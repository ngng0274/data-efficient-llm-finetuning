"""
IFD 상위 40% 내 confidence 분포 분석.

목적: D_new 구성 전, "어렵고(high IFD) + 신뢰도 낮은(low confidence)" 샘플이
     실제로 노이즈성인지 눈으로 확인한다.

Usage:
    python scripts/analyze_ifd_conf_overlap.py
"""

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.alpaca_loader import load_alpaca_all

SCORES_PATH = Path("outputs/alpaca_scores/alpaca_scores.json")
NUM_EVAL    = 1000
SEED        = 42
IFD_TOP_PCT = 0.40
SHOW_N      = 4          # 각 그룹에서 출력할 예시 수


def reconstruct_train_indices(n=52002, seed=42, num_eval=1000):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return set(idx[:num_eval]), idx[num_eval:]   # eval_set, train_list


def pct(label, p25, p50, p75, mn, mx, std):
    return (f"  {label:<30} p25={p25:.4f} p50={p50:.4f} p75={p75:.4f} "
            f"min={mn:.4f} max={mx:.4f} std={std:.4f}")


def main():
    # ── Load ─────────────────────────────────────────────────────────────────
    with open(SCORES_PATH, encoding="utf-8") as f:
        scores_raw = json.load(f)
    score_by_idx = {r["index"]: r for r in scores_raw}

    all_data = load_alpaca_all()   # index-stable

    eval_set, train_list = reconstruct_train_indices()

    # ── Train pool: IFD ≤ 1 and not skipped ──────────────────────────────────
    train_pool = []
    for i in train_list:
        s = score_by_idx.get(i)
        if s is None or s["skipped"] or s["ifd_score"] is None:
            continue
        if s["ifd_score"] > 1.0:
            continue
        if s["combined_conf"] is None:
            continue
        train_pool.append({
            "index":        i,
            "ifd":          s["ifd_score"],
            "global_conf":  s["global_conf"],
            "local_conf":   s["local_conf"],
            "combined_conf": s["combined_conf"],
            "instruction":  all_data[i]["instruction"],
            "input":        all_data[i].get("input", ""),
            "output":       all_data[i]["output"],
        })

    print(f"Train pool (IFD≤1, scored): {len(train_pool)} samples")

    # ── IFD 상위 40% 선택 ────────────────────────────────────────────────────
    pool_sorted_ifd = sorted(train_pool, key=lambda r: r["ifd"])
    cutoff_n = int(len(pool_sorted_ifd) * (1 - IFD_TOP_PCT))
    ifd_top40 = pool_sorted_ifd[cutoff_n:]   # 상위 40% = IFD 높은 쪽

    print(f"IFD top-40% cut  : IFD ≥ {pool_sorted_ifd[cutoff_n]['ifd']:.4f}")
    print(f"IFD top-40% size : {len(ifd_top40)} samples\n")

    # ── IFD top-40% 내 confidence 분포 ───────────────────────────────────────
    ifd_vals  = np.array([r["ifd"]          for r in ifd_top40])
    conf_vals = np.array([r["combined_conf"] for r in ifd_top40])

    print("=" * 70)
    print("[1] IFD top-40% 내 분포")
    print("=" * 70)
    for name, arr in [("IFD score", ifd_vals), ("combined_conf", conf_vals)]:
        print(pct(name,
                  np.percentile(arr, 25), np.percentile(arr, 50),
                  np.percentile(arr, 75), arr.min(), arr.max(), arr.std()))

    # ── Confidence 중앙값으로 상/하 분할 ─────────────────────────────────────
    conf_median = np.median(conf_vals)
    low_conf  = [r for r in ifd_top40 if r["combined_conf"] <  conf_median]
    high_conf = [r for r in ifd_top40 if r["combined_conf"] >= conf_median]

    print(f"\nconf median = {conf_median:.4f}")
    print(f"  conf 하위 50% (노이즈 의심): {len(low_conf)}")
    print(f"  conf 상위 50% (신뢰)       : {len(high_conf)}")

    # ── 두 그룹의 IFD / conf 비교 ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[2] conf 하위 vs 상위 — IFD & conf 평균 비교")
    print("=" * 70)
    for label, grp in [("conf 하위 50% (low )", low_conf),
                       ("conf 상위 50% (high)", high_conf)]:
        ifd_g  = np.array([r["ifd"]          for r in grp])
        conf_g = np.array([r["combined_conf"] for r in grp])
        print(f"  {label}  n={len(grp):<6} "
              f"IFD mean={ifd_g.mean():.4f}±{ifd_g.std():.4f}  "
              f"conf mean={conf_g.mean():.4f}±{conf_g.std():.4f}")

    # ── Conf 하위 샘플 예시 (노이즈 의심) ────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"[3] conf 하위 50% 샘플 예시 (IFD 높고 confidence 낮음 = 노이즈 의심?)")
    print("=" * 70)
    low_sorted = sorted(low_conf, key=lambda r: r["combined_conf"])   # 가장 낮은 것부터
    rng = random.Random(SEED)
    # 최하위 20개 중 랜덤 SHOW_N개
    low_sample = rng.sample(low_sorted[:20], min(SHOW_N, len(low_sorted[:20])))
    for i, r in enumerate(low_sample, 1):
        instr = r["instruction"][:120].replace("\n", " ")
        inp   = r["input"][:60].replace("\n", " ") if r["input"].strip() else "(없음)"
        out   = r["output"][:150].replace("\n", " ")
        print(f"\n  [{i}] index={r['index']}  IFD={r['ifd']:.4f}  conf={r['combined_conf']:.4f}")
        print(f"  instruction : {instr}")
        print(f"  input       : {inp}")
        print(f"  output      : {out}")

    # ── Conf 상위 샘플 예시 (신뢰) ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"[4] conf 상위 50% 샘플 예시 (IFD 높고 confidence 높음 = 어렵지만 신뢰)")
    print("=" * 70)
    high_sorted = sorted(high_conf, key=lambda r: r["combined_conf"], reverse=True)
    high_sample = rng.sample(high_sorted[:20], min(SHOW_N, len(high_sorted[:20])))
    for i, r in enumerate(high_sample, 1):
        instr = r["instruction"][:120].replace("\n", " ")
        inp   = r["input"][:60].replace("\n", " ") if r["input"].strip() else "(없음)"
        out   = r["output"][:150].replace("\n", " ")
        print(f"\n  [{i}] index={r['index']}  IFD={r['ifd']:.4f}  conf={r['combined_conf']:.4f}")
        print(f"  instruction : {instr}")
        print(f"  input       : {inp}")
        print(f"  output      : {out}")

    # ── D_new 예상 구성 미리 보기 ─────────────────────────────────────────────
    n_target = int(len(pool_sorted_ifd) * 0.20)   # 전체의 20%
    d_new_candidates = high_conf   # IFD top-40% ∩ conf 상위 50%
    print("\n" + "=" * 70)
    print(f"[5] D_new 예상 구성 (IFD top-40% ∩ conf 상위 50%)")
    print("=" * 70)
    print(f"  D_new 후보 수         : {len(d_new_candidates)}")
    print(f"  목표 N (전체 20%)     : {n_target}")
    if len(d_new_candidates) >= n_target:
        # 남은 후보 중 IFD 높은 순으로 최종 선택
        d_new_final = sorted(d_new_candidates, key=lambda r: r["ifd"])[-n_target:]
        ifd_final = np.array([r["ifd"]          for r in d_new_final])
        conf_final = np.array([r["combined_conf"] for r in d_new_final])
        print(f"  최종 선택 (IFD 상위) : {len(d_new_final)}")
        print(f"  IFD  range : {ifd_final.min():.4f} – {ifd_final.max():.4f}  mean={ifd_final.mean():.4f}")
        print(f"  conf range : {conf_final.min():.4f} – {conf_final.max():.4f}  mean={conf_final.mean():.4f}")
    else:
        print(f"  WARNING: 후보({len(d_new_candidates)}) < N({n_target}). 파라미터 조정 필요.")

    print("\n분석 완료. 위 예시를 보고 D_new 구성 여부를 결정하세요.")


if __name__ == "__main__":
    main()
