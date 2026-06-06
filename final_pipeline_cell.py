"""
남은 파이프라인 실행 셀
- qwen3 seed 3개 추론 포함 6모델 앙상블
- hardneg qwen25, qwen3
- 최종 8모델 가중 앙상블
"""

import os, sys, gc, json, time, shutil, torch
from pathlib import Path

# ── 환경 설정 ──────────────────────────────────────────
os.chdir('/content/drive/MyDrive/vqa')
sys.path.insert(0, '/content')

DRIVE_DIR = Path('/content/drive/MyDrive/vqa')
SCRIPT    = '/content/train_vqa_ultimate.py'

# ── __main__ 블록 비활성화 후 함수만 로드 ──────────────
print("스크립트 로드 중...")
code = open(SCRIPT).read()
code = code.replace('if __name__ == "__main__":', 'if False:')
_ns = {}
exec(code, _ns)

run_ensemble = _ns["run_ensemble"]
run_hardneg  = _ns["run_hardneg"]
ckpt_dir     = _ns["ckpt_dir"]
print("✅ 함수 로드 완료\n")

# ── 헬퍼 ───────────────────────────────────────────────
def is_done(mt, seed, hardneg=False):
    cd = Path(ckpt_dir(mt, seed, hardneg=hardneg))
    return cd.exists() and (cd / "meta.json").exists()

def get_acc(mt, seed, hardneg=False):
    cd = Path(ckpt_dir(mt, seed, hardneg=hardneg))
    meta = cd / "meta.json"
    return json.loads(meta.read_text()).get("val_acc", 0) if meta.exists() else 0

def save_sub(label):
    dst = str(DRIVE_DIR / f"submission_{label}_{int(time.time())}.csv")
    try:
        shutil.copy("/content/submission_final.csv", dst)
        print(f"  💾 Drive 저장: submission_{label}")
    except Exception as e:
        print(f"  ⚠ Drive 저장 실패: {e}")

def vram_clear():
    gc.collect()
    torch.cuda.empty_cache()
    free = (torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated()) / 1e9
    print(f"  VRAM 여유: {free:.1f}GB")

# ── 현황 출력 ───────────────────────────────────────────
print("=== 체크포인트 현황 ===")
for mt in ["qwen25", "qwen3"]:
    for sd in [42, 123, 777]:
        d = is_done(mt, sd)
        acc = f"acc={get_acc(mt,sd):.4f}" if d else "미완료"
        print(f"  {'✅' if d else '⬜'} {mt}_seed{sd}  {acc}")
    hn = is_done(mt, 42, hardneg=True)
    if hn:
        print(f"  ✅ {mt}_seed42_hn  acc={get_acc(mt,42,True):.4f}")
print()

# ════════════════════════════════════════════════════════
# STEP 1: 6모델 앙상블
# (qwen25 3개 캐시 + qwen3 3개 새 추론)
# ════════════════════════════════════════════════════════
print("▶ STEP 1: 6모델 앙상블")
print("  qwen25 3개: 캐시 로드 (즉시)")
print("  qwen3  3개: 추론 ~3h")
vram_clear()
run_ensemble(use_weighted=False)
save_sub("6model_ensemble")
print("✅ STEP 1 완료\n")

# ════════════════════════════════════════════════════════
# STEP 2: Hard Negative 재학습
# ════════════════════════════════════════════════════════
for mt in ["qwen25", "qwen3"]:
    if is_done(mt, 42, hardneg=True):
        print(f"⏭  {mt} hardneg 완료 → 스킵 (acc={get_acc(mt,42,True):.4f})")
    else:
        print(f"\n▶ STEP 2: {mt} hardneg 재학습 (~5h)")
        vram_clear()
        run_hardneg(mt)
        print(f"✅ {mt} hardneg 완료\n")

# ════════════════════════════════════════════════════════
# STEP 3: 최종 앙상블 (최대 8모델 가중 소프트 보팅)
# (6개 캐시 + hardneg 2개 새 추론 ~2h)
# ════════════════════════════════════════════════════════
print("▶ STEP 3: 최종 앙상블 (가중 소프트 보팅)")
print("  원본 6개: 캐시 로드 (즉시)")
print("  hardneg 2개: 추론 ~2h")
vram_clear()
run_ensemble(use_weighted=True)
save_sub("FINAL")

print("\n" + "="*55)
print("🎉 전체 완료!")
print()
print("=== 최종 현황 ===")
for mt in ["qwen25", "qwen3"]:
    for sd in [42, 123, 777]:
        d = is_done(mt, sd)
        print(f"  {'✅' if d else '⬜'} {mt}_seed{sd}" +
              (f"  acc={get_acc(mt,sd):.4f}" if d else ""))
    hn = is_done(mt, 42, hardneg=True)
    if hn:
        print(f"  ✅ {mt}_seed42_hn  acc={get_acc(mt,42,True):.4f}")
print()
print("제출 파일: /content/drive/MyDrive/vqa/submission_FINAL_xxx.csv")
