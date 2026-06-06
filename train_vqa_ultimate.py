"""
╔══════════════════════════════════════════════════════════════╗
║   VQA 4-Choice — A100 40GB 80시간 풀 공략 최종본              ║
║   목표: 0.98+ accuracy                                       ║
║                                                              ║
║   지원 모델:                                                  ║
║     - Qwen/Qwen2.5-VL-7B-Instruct  (검증된 베이스라인)        ║
║     - Qwen/Qwen3-VL-8B-Instruct    (최신, 성능 우위)          ║
║                                                              ║
║   실행 모드 (RUN_MODE):                                       ║
║     "train"     → 단일 seed 학습                              ║
║     "ensemble"  → 저장된 전체 체크포인트 앙상블+TTA 추론        ║
║     "hardneg"   → 오답 재학습 (hard negative mining)          ║
║     "weighted"  → val acc 기반 가중치 소프트 보팅              ║
║                                                              ║
║   80시간 타임라인:                                             ║
║   [00~18h] Qwen2.5-VL-7B × 3 seed 학습                      ║
║   [18~20h] 1차 앙상블 제출                                     ║
║   [20~38h] Qwen3-VL-8B   × 3 seed 학습                      ║
║   [38~42h] 2차 앙상블 제출                                     ║
║   [42~58h] 오답 Hard Negative 재학습 (양 모델)                  ║
║   [58~80h] 최종 앙상블 튜닝 + 제출                              ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── 설치 명령어 (Colab 첫 셀에서 실행) ──────────────────────────────────────────
# !pip install -q "transformers>=5.0.0" peft "bitsandbytes>=0.46.1" accelerate qwen-vl-utils
# !pip install -q flash-attn --no-build-isolation   # A100 필수 (속도 2배)
# !pip install -q pillow tqdm pandas numpy
# from google.colab import drive; drive.mount('/content/drive')

# ── 런타임 패키지 자동 검증 ─────────────────────────────────────────────────────
import importlib, subprocess, sys

def _ensure(pkg_name, import_name, min_ver, pip_spec):
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "0.0.0")
        from packaging.version import Version
        if Version(ver) >= Version(min_ver):
            print(f"  OK {pkg_name} {ver}")
            return
        print(f"  UP {pkg_name} {ver} -> {min_ver} 업그레이드 중...")
    except ImportError:
        print(f"  IN {pkg_name} 설치 중...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", pip_spec],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    importlib.invalidate_caches()

print("패키지 버전 확인 ...")
_ensure("bitsandbytes", "bitsandbytes", "0.46.1", "bitsandbytes>=0.46.1")
_ensure("transformers",  "transformers",  "5.0.0",  "transformers>=5.0.0")
_ensure("peft",          "peft",          "0.10.0", "peft>=0.10.0")
_ensure("accelerate",    "accelerate",    "0.26.0", "accelerate>=0.26.0")
print("패키지 확인 완료\n")

import os, re, gc, math, random, time, json, warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

warnings.filterwarnings("ignore")
import transformers
print(f"transformers : {transformers.__version__}")
print(f"torch        : {torch.__version__}")

Image.MAX_IMAGE_PIXELS = None

# ════════════════════════════════════════════════════════════════
# ★★★  실행 설정 — 여기만 바꾸세요  ★★★
# ════════════════════════════════════════════════════════════════
RUN_MODE   = "train"         # "train" | "ensemble" | "hardneg" | "weighted"
MODEL_TYPE = "qwen25"        # "qwen25" | "qwen3"
TRAIN_SEED = 42              # 42, 123, 777 순서로 3번 실행

# ════════════════════════════════════════════════════════════════
# 경로 설정
# ════════════════════════════════════════════════════════════════
# ── 경로 설정 ────────────────────────────────────────────────────
DRIVE_DATA_DIR = Path("/content/drive/MyDrive/vqa")

# best 체크포인트: Drive에 저장 (1개 덮어쓰기, 15GB 여유)
LOCAL_CKPT_DIR = Path("/content/drive/MyDrive/vqa/ckpt")
LOCAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_IDS = {
    "qwen25": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3":  "Qwen/Qwen3-VL-8B-Instruct",
}
MODEL_ID = MODEL_IDS[MODEL_TYPE]

def ckpt_dir(model_type, seed, hardneg=False):
    """
    best 저장: Drive/ckpt/{model_type}_seed{seed}
    앙상블 시 각 seed 체크포인트 필요하므로 seed별 유지
    Drive 15GB = 모델 1.5GB × 최대 6개 = 9GB → 가능
    """
    suffix = "_hn" if hardneg else ""
    return str(LOCAL_CKPT_DIR / f"{model_type}_seed{seed}{suffix}")

SUBMISSION_PATH = "/content/submission_final.csv"

# ════════════════════════════════════════════════════════════════
# 하이퍼파라미터 — VRAM에 따라 자동 조정
# ════════════════════════════════════════════════════════════════
import torch as _torch
_vram_gb = _torch.cuda.get_device_properties(0).total_memory / 1e9            if _torch.cuda.is_available() else 0
print(f"VRAM 감지: {_vram_gb:.0f}GB → ", end="")

if _vram_gb >= 70:          # A100 80GB
    BATCH_SIZE     = 2
    GRAD_ACCUM     = 4      # effective batch = 8 (step 수 절반)
    LORA_R         = 128
    LORA_ALPHA     = 256
    print("80GB 모드 (r=128, batch=2, accum=4)")
else:                       # A100 40GB
    BATCH_SIZE     = 1
    GRAD_ACCUM     = 4      # effective batch = 4
    LORA_R         = 64
    LORA_ALPHA     = 128
    print("40GB 모드 (r=64, batch=1)")

SEEDS          = [42, 123, 777]
EPOCHS         = 3           # 8500샘플 VLM fine-tune: 3ep 이후 오버핏 위험
EARLY_STOP_PAT = 2
LR             = 1e-4
WARMUP_RATIO   = 0.05
LORA_DROPOUT   = 0.05
VALID_RATIO    = 0.1     
TIME_LIMIT_HR  = 22.0
MAX_NEW_TOKENS = 5
LABEL_SMOOTH   = 0.1
TTA_N          = 1           # 4→1: 데드라인 내 완료 (TTA 없이 원본만)
HARDNEG_EPOCHS = 2           # hardneg 6 seed로 확장됐으므로 ep 축소로 시간 상쇄
DEV_CONF_THRESHOLD = 0.8   # dev 데이터 신뢰도 필터

# 해상도: 두 환경 모두 동일
IMAGE_MIN_PIX = 256 * 28 * 28    # ~200K
IMAGE_MAX_PIX = 1280 * 28 * 28   # ~1M

# ════════════════════════════════════════════════════════════════
# 시드 / 디바이스
# ════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(TRAIN_SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"GPU  : {torch.cuda.get_device_name()}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# ════════════════════════════════════════════════════════════════
# 데이터 로드
# ════════════════════════════════════════════════════════════════
# Drive에서 CSV 로드
_csv_base = DRIVE_DATA_DIR
train_df = pd.read_csv(_csv_base / "train.csv" if (_csv_base/"train.csv").exists() else "train.csv")
test_df  = pd.read_csv(_csv_base / "test.csv"  if (_csv_base/"test.csv").exists()  else "test.csv")

# dev 데이터 병합 (신뢰도 필터 적용)
_dev_path = _csv_base / "dev.csv" if (_csv_base/"dev.csv").exists() else Path("dev.csv")
if _dev_path.exists():
    from collections import Counter as _Counter
    _dev = pd.read_csv(_dev_path)
    _ans_cols = ["answer1","answer2","answer3","answer4","answer5"]

    def _majority(row):
        votes = [str(row[c]).strip().lower() for c in _ans_cols
                 if pd.notna(row[c]) and str(row[c]).strip() not in ("nan","")]
        if not votes: return ("a", 0.0)
        cnt = _Counter(votes)
        top, top_n = cnt.most_common(1)[0]
        return (top, top_n / len(votes))

    _res              = _dev.apply(_majority, axis=1)
    _dev["answer"]    = _res.apply(lambda x: x[0])
    _dev["confidence"]= _res.apply(lambda x: x[1])

    # 신뢰도 필터링
    _dev_filtered = _dev[_dev["confidence"] >= DEV_CONF_THRESHOLD].copy()
    _dev_filtered = _dev_filtered[train_df.columns]  # train과 동일 컬럼만

    train_df = pd.concat([train_df, _dev_filtered], ignore_index=True)
    print(f"dev 데이터 추가: {len(_dev_filtered)}개 (confidence>={DEV_CONF_THRESHOLD}) "
          f"→ 전체 train: {len(train_df)}개")
else:
    print(f"dev.csv 없음 → train만 사용 ({len(train_df)}개)")

# NaN 처리 (train_2125 → 'd')
nan_mask = train_df["answer"].isna()
if nan_mask.any():
    print(f"⚠  NaN answer {nan_mask.sum()}개 → 'd' 처리")
    train_df.loc[nan_mask, "answer"] = "d"

print(f"Answer 분포: {train_df['answer'].value_counts().sort_index().to_dict()}")

def make_split(seed: int):
    """seed별로 다르게 셔플 → 각 모델이 다른 valid 경험"""
    df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    sp = int(len(df) * (1 - VALID_RATIO))
    return df.iloc[:sp].reset_index(drop=True), df.iloc[sp:].reset_index(drop=True)

train_sub, valid_sub = make_split(TRAIN_SEED)
print(f"[seed={TRAIN_SEED}] Train:{len(train_sub)}  Valid:{len(valid_sub)}  Test:{len(test_df)}")

# ════════════════════════════════════════════════════════════════
# 이미지 경로
# ════════════════════════════════════════════════════════════════
def get_img_path(row_path: str) -> str:
    p = Path(row_path)
    if p.exists(): return str(p)
    # Drive 경로 우선 탐색
    for c in [
        DRIVE_DATA_DIR / p,           # Drive/vqa/train/xxx.jpg
        DRIVE_DATA_DIR / p.name,      # Drive/vqa/xxx.jpg
        DRIVE_DATA_DIR / "train" / p.name,
        DRIVE_DATA_DIR / "test"  / p.name,
        Path(".") / p,
        Path("train") / p.name,
        Path("test")  / p.name,
        Path(p.name),
    ]:
        if c.exists(): return str(c)
    return str(p)

# ════════════════════════════════════════════════════════════════
# 프롬프트
# ════════════════════════════════════════════════════════════════
SYSTEM_INSTRUCT = (
    "You are a recycling classification assistant. "
    "You MUST answer with ONLY one single lowercase letter: a, b, c, or d. "
    "Do NOT write words, explanations, or the full option. "
    "Output ONLY the letter itself, nothing else."
)

def build_prompt(q, a, b, c, d) -> str:
    # 질문 유형별 힌트 (오답 패턴: 개수 52%, 재질 22%, 재활용 15%)
    if any(kw in q for kw in ['몇 개', '몇개', '개수', '몇 병', '몇 장',
                                '몇 캔', '몇 통', '몇 봉', '몇 박스']):
        hint = "이미지 속 물체를 하나씩 천천히 세어서 정확한 개수를 확인하세요.\n"
    elif any(kw in q for kw in ['재질', '소재', '재료', '만들어', '무슨 재질']):
        hint = "이미지 속 물체의 재질(플라스틱/유리/금속/종이 등)을 주의 깊게 관찰하세요.\n"
    elif any(kw in q for kw in ['재활용', '분류', '어떻게 버려', '어느 통']):
        hint = "재활용 분류 기준(종이류/플라스틱류/유리병류/금속류)을 기반으로 판단하세요.\n"
    elif any(kw in q for kw in ['색', '색깔', '색상', '무슨 색']):
        hint = "이미지 속 물체의 색깔을 정확히 관찰하세요.\n"
    else:
        hint = ""
    return (
        f"{hint}{q}\n"
        f"(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
        "Answer: (output only one letter — a, b, c, or d)\n"
        "정답 (반드시 a/b/c/d 중 딱 한 글자만):"
    )

# ════════════════════════════════════════════════════════════════
# 데이터 증강
# ════════════════════════════════════════════════════════════════
def augment(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """strength: 0=원본, 1=보통, 2=강함"""
    if strength == 0:
        return img
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.4 * strength:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.3 * strength:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.3 * strength:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.25 * strength:
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 2.0))
    if random.random() < 0.3 * strength:
        angle = random.uniform(-12, 12)
        img = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))
    return img

# ════════════════════════════════════════════════════════════════
# Dataset / Collator
# ════════════════════════════════════════════════════════════════
class VQADataset(Dataset):
    def __init__(self, df, processor, is_train=True, aug_strength=1.0):
        self.df          = df.reset_index(drop=True)
        self.processor   = processor
        self.is_train    = is_train
        self.aug_strength= aug_strength

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(get_img_path(str(row["path"]))).convert("RGB")
        if self.is_train:
            img = augment(img, self.aug_strength)

        text = build_prompt(str(row["question"]),
                            str(row["a"]), str(row["b"]),
                            str(row["c"]), str(row["d"]))
        msgs = [
            {"role": "system", "content": [{"type": "text",  "text": SYSTEM_INSTRUCT}]},
            {"role": "user",   "content": [{"type": "image", "image": img},
                                           {"type": "text",  "text": text}]},
        ]
        if self.is_train:
            gold = str(row["answer"]).strip().lower()
            msgs.append({"role": "assistant",
                         "content": [{"type": "text", "text": gold}]})
        return {"messages": msgs, "image": img}


@dataclass
class VQACollator:
    processor: Any
    is_train:  bool = True

    def __call__(self, batch):
        texts, images = [], []
        # Qwen3: thinking 토큰이 학습 label에 포함되지 않도록 비활성화
        _is_qwen3 = 'Qwen3' in getattr(self.processor.tokenizer, 'name_or_path', '')
        _tmpl_kw  = {"tokenize": False, "add_generation_prompt": False}
        if _is_qwen3:
            _tmpl_kw["enable_thinking"] = False
        for s in batch:
            texts.append(self.processor.apply_chat_template(s["messages"], **_tmpl_kw))
            images.append(s["image"])
        enc = self.processor(text=texts, images=images,
                             padding=True, return_tensors="pt")
        if self.is_train:
            labels = enc["input_ids"].clone()
            # 프롬프트 토큰 마스킹 → 정답 토큰만 loss 계산 (0.93 방식)
            asst_str = "<|im_start|>assistant" + chr(10)
            assistant_token = self.processor.tokenizer.encode(
                asst_str, add_special_tokens=False)
            for i in range(len(texts)):
                input_ids = enc["input_ids"][i].tolist()
                mask_end = len(input_ids)
                for j in range(len(input_ids) - len(assistant_token)):
                    if input_ids[j:j+len(assistant_token)] == assistant_token:
                        mask_end = j + len(assistant_token)
                        break
                labels[i, :mask_end] = -100
            enc["labels"] = labels
        return enc

# ════════════════════════════════════════════════════════════════
# 헬퍼
# ════════════════════════════════════════════════════════════════
def extract_choice(text: str) -> str:
    """
    다양한 출력 형태에서 a/b/c/d 추출
    - "c"                → c
    - "c\n"              → c
    - "(c) 플라스틱"       → c
    - "정답: c"           → c
    - "Answer: c"        → c
    - "<think>...</think>c" → c  (Qwen3 thinking 잔재)
    """
    # 1. thinking 블록 제거 (Qwen3)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip().lower()

    # 2. 첫 글자가 바로 정답인 경우 (가장 이상적)
    if text and text[0] in "abcd" and (len(text) == 1 or not text[1].isalpha()):
        return text[0]

    # 3. "answer: c" / "정답: c" / "(c)" 패턴
    for pat in [
        r"answer\s*:\s*([abcd])",
        r"정답\s*[:：]\s*([abcd])",
        r"\(([abcd])\)",
        r"^([abcd])[^\w]",
        r"[^a-z]([abcd])[^a-z]",
        r"([abcd])",
    ]:
        m = re.search(pat, text)
        if m: return m.group(1)

    # 4. 줄별 역순 탐색
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line in ("a","b","c","d"): return line
        for tok in line.split():
            tok = tok.strip("().,:")
            if tok in ("a","b","c","d"): return tok

    return "a"  # fallback

def ls_loss(logits, labels, smoothing=0.1):
    """사용 안 함 — 모델 기본 loss 사용"""
    pass

# ════════════════════════════════════════════════════════════════
# BnB 설정
# ════════════════════════════════════════════════════════════════
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)

# ════════════════════════════════════════════════════════════════
# 모델 로드 (Qwen2.5-VL / Qwen3-VL 자동 분기)
# ════════════════════════════════════════════════════════════════
def load_model_and_processor(model_id: str, for_train: bool = True):
    """모델 타입에 따라 올바른 클래스로 로드 (transformers 5.0 기준)"""
    print(f"\n모델 로드: {model_id}")

    # transformers 5.0: AutoModelForVision2Seq 삭제 → 직접 클래스 사용
    if "Qwen3" in model_id:
        from transformers import Qwen3VLForConditionalGeneration
        ModelClass = Qwen3VLForConditionalGeneration
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration
        ModelClass = Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=IMAGE_MIN_PIX,
        max_pixels=IMAGE_MAX_PIX,
        trust_remote_code=True,
    )

    # flash_attention_2: A100에서 속도 2배, VRAM 절감
    try:
        base = ModelClass.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        print("✓ FlashAttention-2 활성화")
    except Exception:
        base = ModelClass.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("⚠ FlashAttention-2 미사용 (설치 안 됨)")

    if for_train:
        base = prepare_model_for_kbit_training(base)
        # gradient checkpointing: 80GB VRAM이면 불필요, 오히려 버그 유발
        # base.gradient_checkpointing_enable(...)  ← 비활성화
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            bias="none",
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                             "gate_proj","up_proj","down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
        model.print_trainable_parameters()
    else:
        model = base

    return model, processor

def load_trained_model(model_type: str, seed: int, ckpt_path: str = None):
    """저장된 체크포인트 로드 (추론용)"""
    cd = ckpt_path if ckpt_path else ckpt_dir(model_type, seed)
    model_id = MODEL_IDS[model_type]
    print(f"\n추론 모델 로드: {cd}")

    # transformers 5.0: 직접 클래스 사용
    if "Qwen3" in model_id:
        from transformers import Qwen3VLForConditionalGeneration as MC
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration as MC

    try:
        base = MC.from_pretrained(
            model_id, quantization_config=bnb_cfg, device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
    except Exception:
        base = MC.from_pretrained(
            model_id, quantization_config=bnb_cfg, device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model = PeftModel.from_pretrained(base, cd)
    model.eval()

    proc = AutoProcessor.from_pretrained(
        cd, min_pixels=IMAGE_MIN_PIX, max_pixels=IMAGE_MAX_PIX,
        trust_remote_code=True,
    )
    return model, proc

# ════════════════════════════════════════════════════════════════
# 로짓 기반 추론 — generate() 미사용으로 5~10× 속도 향상
# ════════════════════════════════════════════════════════════════




@torch.no_grad()
@torch.no_grad()
def infer_one(model, proc, row, aug_strength=0) -> str:
    """generate() 기반 추론 — 로컬 0.93 검증 방식"""
    img  = Image.open(get_img_path(str(row["path"]))).convert("RGB")
    img  = augment(img, aug_strength)
    text = build_prompt(str(row["question"]),
                        str(row["a"]), str(row["b"]),
                        str(row["c"]), str(row["d"]))
    msgs = [
        {"role": "system", "content": [{"type": "text",  "text": SYSTEM_INSTRUCT}]},
        {"role": "user",   "content": [{"type": "image", "image": img},
                                       {"type": "text",  "text": text}]},
    ]
    _tmpl_kw = {"tokenize": False, "add_generation_prompt": True}
    if "Qwen3" in getattr(proc.tokenizer, "name_or_path", ""):
        _tmpl_kw["enable_thinking"] = False
    tin = proc.apply_chat_template(msgs, **_tmpl_kw)
    inp = proc(text=[tin], images=[img], return_tensors="pt").to(device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(
            **inp,
            max_new_tokens=5,
            do_sample=False,
            eos_token_id=proc.tokenizer.eos_token_id,
        )
    gen = out[:, inp["input_ids"].shape[1]:]
    return extract_choice(proc.batch_decode(gen, skip_special_tokens=True)[0])


def infer_tta(model, proc, row):
    """
    TTA_N=1: 원본 1회 추론 → (pred, uniform_probs) 반환
    TTA_N>1: 원본 2회 + 증강 (N-2)회 → 다수결 → (pred, vote_probs) 반환
    항상 (str, np.ndarray[4]) 형태로 반환
    """
    CHOICE_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}

    if TTA_N == 1:
        pred = infer_one(model, proc, row, 0)
        probs = np.zeros(4)
        probs[CHOICE_MAP.get(pred, 0)] = 1.0
        return pred, probs

    votes = [
        infer_one(model, proc, row, 0),
        infer_one(model, proc, row, 0),
    ]
    for _ in range(TTA_N - 2):
        votes.append(infer_one(model, proc, row, 1))

    cnt = Counter(votes)
    pred = cnt.most_common(1)[0][0]
    probs = np.array([cnt.get(c, 0) / len(votes) for c in "abcd"])
    return pred, probs


# ════════════════════════════════════════════════════════════════
# Validation
# ════════════════════════════════════════════════════════════════
def validate(model, proc, seed: int) -> tuple[float, pd.DataFrame]:
    model.eval()
    _, vsub = make_split(seed)
    correct, wrong_rows = 0, []

    for i in tqdm(range(len(vsub)), desc="  [valid]", leave=False):
        row  = vsub.iloc[i]
        pred = infer_one(model, proc, row, 0)
        gold = str(row["answer"]).strip().lower()
        correct += int(pred == gold)
        if pred != gold:
            wrong_rows.append({**row.to_dict(), "pred": pred})

    acc      = correct / len(vsub)
    wrong_df = pd.DataFrame(wrong_rows)
    if len(wrong_df):
        wp = LOCAL_CKPT_DIR / f"wrong_{MODEL_TYPE}_seed{seed}.csv"
        wrong_df.to_csv(wp, index=False)
        print(f"  오답 {len(wrong_df)}개 → {wp}")
    model.train()
    return acc, wrong_df

# ════════════════════════════════════════════════════════════════
# PHASE 1/2 — 학습
# ════════════════════════════════════════════════════════════════
def run_training(model_type: str, seed: int, extra_df: Optional[pd.DataFrame] = None,
                 is_hardneg: bool = False):
    """
    extra_df: Hard Negative 재학습 시 오답 샘플 추가 DataFrame
    is_hardneg: True면 _hn 폴더에 저장 (원본 체크포인트 보존)
    """
    set_seed(seed)
    save_dir    = ckpt_dir(model_type, seed, hardneg=is_hardneg)
    model_id    = MODEL_IDS[model_type]
    t0          = time.time()

    print(f"\n{'='*60}")
    print(f" 학습: {model_type}  seed={seed}")
    print(f" 모델: {model_id}")
    print(f" 저장: {save_dir}")
    print(f" 해상도: {IMAGE_MIN_PIX//1000}K~{IMAGE_MAX_PIX//1000}K px")
    print(f" LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    if extra_df is not None:
        print(f" Hard Negative 추가: {len(extra_df)}개")
    print(f"{'='*60}\n")

    # Drive에 체크포인트 있으면 학습 스킵
    if Path(save_dir).exists() and extra_df is None:
        meta_path = Path(save_dir) / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            print(f"  ✅ Drive 체크포인트 존재 (val_acc={meta.get('val_acc',0):.4f}) → 학습 스킵")
            return meta.get("val_acc", 0.9)

    model, proc = load_model_and_processor(model_id, for_train=True)

    tr_sub, va_sub = make_split(seed)

    # Hard Negative 있으면 train set에 3배 오버샘플링 후 합치기
    if extra_df is not None and len(extra_df):
        extra_3x = pd.concat([extra_df] * 3, ignore_index=True)
        tr_sub   = pd.concat([tr_sub, extra_3x], ignore_index=True)
        print(f"  오답 오버샘플링 후 train: {len(tr_sub)}개")

    ds     = VQADataset(tr_sub, proc, is_train=True, aug_strength=1.0)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=VQACollator(proc, True),
                        num_workers=4, pin_memory=True)

    model.to(device)
    opt   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01, betas=(0.9, 0.999),
    )
    n_ep  = HARDNEG_EPOCHS if extra_df is not None else EPOCHS
    total = n_ep * math.ceil(len(loader) / GRAD_ACCUM)
    warm  = max(1, int(total * WARMUP_RATIO))
    sched = get_cosine_schedule_with_warmup(opt, warm, total)
    scaler= torch.amp.GradScaler("cuda", enabled=True)

    print(f"스텝 총계: {total}  Warmup: {warm}")
    print(f"Effective batch: {BATCH_SIZE*GRAD_ACCUM}  LR: {LR}\n")

    best_acc = -1.0; no_imp = 0

    for epoch in range(1, n_ep + 1):
        if (time.time()-t0)/3600 >= TIME_LIMIT_HR:
            print("⚠  시간 초과"); break

        model.train(); opt.zero_grad(set_to_none=True); running = 0.0
        pbar = tqdm(enumerate(loader, 1), total=len(loader),
                    desc=f"[{model_type} s{seed}] E{epoch}/{n_ep}", unit="b")

        for step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out  = model(**batch)
                loss = out.loss / GRAD_ACCUM   # 모델 기본 loss (검증된 방식)

            scaler.scale(loss).backward()
            running += loss.item()

            if step % GRAD_ACCUM == 0 or step == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sched.step()
                denom = step % GRAD_ACCUM or GRAD_ACCUM
                pbar.set_postfix({
                    "loss": f"{running/denom*GRAD_ACCUM:.4f}",
                    "lr":   f"{sched.get_last_lr()[0]:.2e}",
                    "hr":   f"{(time.time()-t0)/3600:.1f}h",
                })
                running = 0.0

            if (time.time()-t0)/3600 >= TIME_LIMIT_HR:
                print("\n⚠  배치 중 시간 초과"); break

        elapsed = (time.time()-t0)/60
        print(f"\n[{model_type} s{seed} | E{epoch}] {elapsed:.1f}min")

        val_acc, _ = validate(model, proc, seed)
        print(f"  Valid Acc: {val_acc:.4f}  (best: {best_acc:.4f})")

        # epoch 체크포인트: Colab 로컬에만 저장 (Drive 용량 절약)
        # best만 Drive에 저장됨
        epoch_dir = f"/content/ckpt_epoch/{model_type}_seed{seed}_ep{epoch}"
        Path(epoch_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        proc.save_pretrained(epoch_dir)
        print(f"  💾 epoch 체크포인트 → {epoch_dir} (로컬)")

        if val_acc > best_acc:
            best_acc = val_acc; no_imp = 0
            model.save_pretrained(save_dir)
            proc.save_pretrained(save_dir)
            meta = {"model_type": model_type, "seed": seed, "val_acc": val_acc,
                    "epoch": epoch, "elapsed_min": (time.time()-t0)/60}
            (Path(save_dir) / "meta.json").write_text(json.dumps(meta))
            print(f"  ✓ Best 갱신  acc={best_acc:.4f} → {save_dir}")
        else:
            no_imp += 1
            print(f"  – No improvement ({no_imp}/{EARLY_STOP_PAT})")
            if no_imp >= EARLY_STOP_PAT:
                print("  Early stopping!"); break

        model.train()

    print(f"\n=== {model_type} seed={seed} 완료: {(time.time()-t0)/60:.1f}min | Best: {best_acc:.4f} ===")
    del model; gc.collect(); torch.cuda.empty_cache()
    return best_acc

# ════════════════════════════════════════════════════════════════
# PHASE 3 — Hard Negative Mining 재학습
# ════════════════════════════════════════════════════════════════
def run_hardneg(model_type: str):
    """
    저장된 best 체크포인트로 val 추론 → 오답 수집
    → 오답 3배 오버샘플링 후 재학습 (hn_ prefix로 저장)
    """
    print(f"\n{'='*60}")
    print(f" Hard Negative Mining: {model_type}")
    print(f"{'='*60}")

    all_wrong = []
    for seed in SEEDS:
        cd = ckpt_dir(model_type, seed)
        if not Path(cd).exists():
            print(f"  seed={seed} 체크포인트 없음, 스킵")
            continue

        model, proc = load_trained_model(model_type, seed)
        _, vsub = make_split(seed)

        wrong = []
        for i in tqdm(range(len(vsub)), desc=f"  오답 수집 seed={seed}", leave=False):
            row  = vsub.iloc[i]
            pred = infer_one(model, proc, row, 0)
            gold = str(row["answer"]).strip().lower()
            if pred != gold:
                wrong.append(row.to_dict())

        print(f"  seed={seed}: 오답 {len(wrong)}개 / {len(vsub)}개")
        all_wrong.extend(wrong)
        del model; gc.collect(); torch.cuda.empty_cache()

    if not all_wrong:
        print("  오답 없음 — Hard Negative 재학습 스킵")
        return

    wrong_df = pd.DataFrame(all_wrong).drop_duplicates(subset=["id"])
    print(f"\n  전체 오답(중복 제거): {len(wrong_df)}개")
    wrong_df.to_csv(str(LOCAL_CKPT_DIR / f"hardneg_{model_type}.csv"), index=False)

    # 개수 문제 오답 5배 오버샘플링 (오답의 52~57%가 개수 문제)
    count_kws = ['몇 개', '몇개', '개수', '몇 병', '몇 장', '몇 캔', '몇 통', '몇 봉', '몇 박스']
    count_mask   = wrong_df['question'].apply(lambda q: any(kw in str(q) for kw in count_kws))
    count_wrong  = wrong_df[count_mask]
    others_wrong = wrong_df[~count_mask]

    if len(count_wrong) > 0:
        extra_df = pd.concat(
            [others_wrong] * 3 + [count_wrong] * 5,
            ignore_index=True
        )
        print(f"  오버샘플링: 개수오답 {len(count_wrong)}개×5 + 기타오답 {len(others_wrong)}개×3")
    else:
        extra_df = pd.concat([wrong_df] * 3, ignore_index=True)
        print(f"  전체 3배 오버샘플링")

    # 재학습: seed=42, hardneg 전용 폴더(_hn)에 저장
    run_training(model_type, seed=42, extra_df=extra_df, is_hardneg=True)

# ════════════════════════════════════════════════════════════════
# PHASE 4 — 앙상블 + TTA 추론
# ════════════════════════════════════════════════════════════════
def _load_or_infer(ck: dict) -> tuple:
    """
    Drive에 이미 추론 결과(sub_xxx.csv)가 있으면 로드,
    없으면 모델 로드 후 추론 → 저장
    반환: (val_acc, [(pred, probs), ...])
    """
    mt    = ck["model_type"]
    sd    = ck["seed"]
    label = ck.get("label", mt)
    suffix = "_hn" if "_hn" in label else ""
    sub_path = LOCAL_CKPT_DIR / f"sub_{mt}_s{sd}{suffix}.csv"

    # 캐시 있으면 로드
    if sub_path.exists():
        print(f"  ✅ 캐시 로드: {sub_path.name}")
        cached = pd.read_csv(sub_path)
        # probs는 균등(캐시엔 없음) → 하드보팅용으로 1.0 할당
        preds = [(row["answer"], np.eye(4)["abcd".index(row["answer"])])
                 for _, row in cached.iterrows()]
        return ck["val_acc"], preds

    # 없으면 추론
    print(f"\n--- {label} seed={sd} 추론 ---")
    model, proc = load_trained_model(mt, sd, ckpt_path=ck["ckpt_dir"])
    preds = []
    for i in tqdm(range(len(test_df)), desc=f"{label}_s{sd}"):
        pred, probs = infer_tta(model, proc, test_df.iloc[i])
        preds.append((pred, probs))

    # 저장
    hard_preds = [p for p, _ in preds]
    pd.DataFrame({"id": test_df["id"], "answer": hard_preds}).to_csv(sub_path, index=False)
    print(f"  💾 저장: {sub_path.name}  분포: {Counter(hard_preds)}")
    del model; gc.collect(); torch.cuda.empty_cache()
    return ck["val_acc"], preds


def run_ensemble(use_weighted: bool = False):
    """
    use_weighted=False: 다수결 (hard voting)
    use_weighted=True:  val_acc 가중 소프트 보팅

    ★ 이미 추론한 모델은 캐시(sub_xxx.csv)에서 즉시 로드
      → 2차 앙상블: qwen3만 새로 추론, qwen25는 캐시 사용
      → 최종 앙상블: hardneg만 새로 추론, 나머지 캐시 사용
    """
    print(f"\n{'='*60}")
    print(f" 앙상블 추론  (weighted={use_weighted}, TTA×{TTA_N})")
    print(f"{'='*60}")

    # 사용 가능한 체크포인트 수집 (원본 + hardneg _hn 포함)
    ckpts = []
    for mt in ["qwen25", "qwen3"]:
        for sd in SEEDS:
            for hn in [False, True]:
                cd = Path(ckpt_dir(mt, sd, hardneg=hn))
                if cd.exists():
                    meta_path = cd / "meta.json"
                    val_acc = json.loads(meta_path.read_text())["val_acc"] \
                              if meta_path.exists() else 0.9
                    ckpts.append({"model_type": mt, "seed": sd,
                                   "ckpt_dir": str(cd), "val_acc": val_acc,
                                   "label": f"{mt}_hn" if hn else mt})

    if not ckpts:
        raise RuntimeError("체크포인트 없음. 먼저 train 실행.")
    print(f"사용 체크포인트 {len(ckpts)}개:")
    for c in ckpts:
        suffix = "_hn" if "_hn" in c.get("label","") else ""
        sub_path = LOCAL_CKPT_DIR / f"sub_{c['model_type']}_s{c['seed']}{suffix}.csv"
        cached = "캐시 ✅" if sub_path.exists() else "추론 필요"
        print(f"  {c.get('label', c['model_type'])} seed={c['seed']}  "
              f"val_acc={c['val_acc']:.4f}  [{cached}]")

    # 캐시 우선 로드 or 추론
    all_preds = []
    for ck in ckpts:
        val_acc, preds = _load_or_infer(ck)
        all_preds.append((val_acc, preds))

    # ── 최종 앙상블 ───────────────────────────────────────────────
    final = []
    agree = 0
    for i in range(len(test_df)):
        if use_weighted:
            total = np.zeros(4)
            for w, preds in all_preds:
                total += w * preds[i][1]
            winner = "abcd"[int(total.argmax())]
        else:
            votes  = [preds[i][0] for _, preds in all_preds]
            cnt    = Counter(votes)
            winner = cnt.most_common(1)[0][0]
            if cnt.most_common(1)[0][1] == len(all_preds): agree += 1
        final.append(winner)

    n = len(final)
    if not use_weighted:
        print(f"\n완전 동의: {agree} ({agree/n*100:.1f}%)")

    sub = pd.DataFrame({"id": test_df["id"], "answer": final})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n✅  {SUBMISSION_PATH}")
    print(sub["answer"].value_counts().sort_index())

    try:
        sub.to_csv(str(DRIVE_DATA_DIR / "submission_final.csv"), index=False)
        print(f"  💾 submission → Drive 저장 완료")
    except Exception:
        print(f"  ⚠  Drive 저장 실패")

# ════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()

    if RUN_MODE == "train":
        print(f"\n◆ train  |  model={MODEL_TYPE}  seed={TRAIN_SEED}")
        print(f"◆ 다음: seed 42→123→777 순서로 반복, 그 후 RUN_MODE='ensemble'")
        run_training(MODEL_TYPE, TRAIN_SEED)
        next_s = [s for s in SEEDS if s != TRAIN_SEED
                  and not Path(ckpt_dir(MODEL_TYPE, s)).exists()]
        if next_s:
            print(f"\n⏭  다음: TRAIN_SEED = {next_s[0]}")
        else:
            print(f"\n⏭  모든 seed 완료 → RUN_MODE = 'ensemble'")

    elif RUN_MODE == "hardneg":
        print(f"\n◆ hardneg  |  model={MODEL_TYPE}")
        run_hardneg(MODEL_TYPE)

    elif RUN_MODE == "ensemble":
        print(f"\n◆ ensemble (hard voting + TTA)")
        run_ensemble(use_weighted=False)

    elif RUN_MODE == "weighted":
        print(f"\n◆ weighted ensemble (soft voting + TTA)")
        run_ensemble(use_weighted=True)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}min ({(time.time()-t0)/3600:.2f}h)")