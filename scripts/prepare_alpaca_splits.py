"""
Construct A/B/C/D train datasets from Alpaca scores.

Input:
  outputs/alpaca_scores/alpaca_scores.json   — IFD + confidence scores (index = original order)
  reference/Cherry_LLM/data/alpaca_data.json — original text (instruction/input/output)

Train/eval split:
  Reproduced with the same seed as load_alpaca(num_eval=1000, seed=42):
    shuffle all 52002 indices → first 1000 = eval, remaining 51002 = train

Selection ratio:
  N = int(51002 × 0.20) = 10200  (20% of all train, consistent across B/C/D)

Datasets:
  A  full     51002  all train samples
  B  random   10200  random 20% of all train (seed=42)
  C  ifd      10200  IFD top-20% from IFD≤1 pool
  D  conf     10200  Combined-confidence top-20% from IFD≤1 pool

Output layout:
  outputs/alpaca_splits/eval.json            shared eval set (1000)
  outputs/alpaca_A_full/train.json
  outputs/alpaca_B_random/train.json
  outputs/alpaca_C_ifd/train.json
  outputs/alpaca_D_confidence/train.json
  outputs/alpaca_splits/split_info.json      selection summary

Usage:
    python scripts/prepare_alpaca_splits.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.alpaca_loader import load_alpaca_all


SCORES_PATH  = Path("outputs/alpaca_scores/alpaca_scores.json")
SPLITS_DIR   = Path("outputs/alpaca_splits")
DIRS = {
    "A": Path("outputs/alpaca_A_full"),
    "B": Path("outputs/alpaca_B_random"),
    "C": Path("outputs/alpaca_C_ifd"),
    "D": Path("outputs/alpaca_D_confidence"),
}
NUM_EVAL  = 1000
SEED      = 42
SEL_RATIO = 0.20


def save_dataset(samples, path: Path, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Alpaca format: keep only instruction/input/output
    records = [{"instruction": s["instruction"],
                "input":       s["input"],
                "output":      s["output"]} for s in samples]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"  {label:<22} → {path}  ({len(records)} samples)")
    return len(records)


def main():
    # ── Load scores ───────────────────────────────────────────────────────────
    if not SCORES_PATH.exists():
        print(f"ERROR: {SCORES_PATH} not found. Run compute_alpaca_scores.py first.")
        sys.exit(1)

    with open(SCORES_PATH, encoding="utf-8") as f:
        scores_raw = json.load(f)

    # Index-keyed lookup: index → score record
    score_by_idx = {r["index"]: r for r in scores_raw}
    print(f"Loaded {len(scores_raw)} score records from {SCORES_PATH}")

    # ── Load original text ────────────────────────────────────────────────────
    print(f"Loading Alpaca original data ...")
    all_data = load_alpaca_all()        # 52002, index i = all_data[i]
    print(f"  {len(all_data)} samples loaded")

    # ── Reproduce train/eval split (must match load_alpaca seed exactly) ──────
    n = len(all_data)                   # 52002
    idx_shuffled = list(range(n))
    rng = random.Random(SEED)
    rng.shuffle(idx_shuffled)

    eval_indices  = set(idx_shuffled[:NUM_EVAL])
    train_indices = idx_shuffled[NUM_EVAL:]     # ordered list, 51002

    N = int(len(train_indices) * SEL_RATIO)     # 10200
    print(f"\nSplit  : {len(train_indices)} train / {len(eval_indices)} eval  (seed={SEED})")
    print(f"N (20%): {N}")

    # ── Attach score + data to each train index ───────────────────────────────
    # Preserve train_indices order (post-shuffle) for reproducibility
    train_rows = []
    n_missing_score = 0
    for i in train_indices:
        d = all_data[i]
        s = score_by_idx.get(i, {})
        if not s:
            n_missing_score += 1
        train_rows.append({
            "index":        i,
            "instruction":  d["instruction"],
            "input":        d["input"],
            "output":       d["output"],
            "ifd_score":    s.get("ifd_score"),
            "combined_conf": s.get("combined_conf"),
            "skipped_score": s.get("skipped", True),
        })

    if n_missing_score:
        print(f"  WARNING: {n_missing_score} train samples have no score record")

    # ── Pool: train samples where IFD is valid and IFD ≤ 1 ──────────────────
    pool = [r for r in train_rows
            if r["ifd_score"] is not None and r["ifd_score"] <= 1.0]
    print(f"\nIFD≤1 pool (from train): {len(pool)} samples")
    if len(pool) < N:
        print(f"  WARNING: pool ({len(pool)}) < N ({N}). Will use entire pool for C/D.")

    # ── Dataset A: all train ──────────────────────────────────────────────────
    dataset_A = train_rows          # all 51002

    # ── Dataset B: random 20% of all train ───────────────────────────────────
    rng_b = random.Random(SEED)
    dataset_B = rng_b.sample(train_rows, N)

    # ── Dataset C: IFD top-20% from pool ─────────────────────────────────────
    # Cherry LLM: sort ascending → take LAST N  (= highest IFD values)
    pool_sorted_ifd = sorted(pool, key=lambda r: r["ifd_score"])
    dataset_C = pool_sorted_ifd[-N:] if len(pool) >= N else pool_sorted_ifd

    # ── Dataset D: Combined-confidence top-20% from pool ─────────────────────
    pool_with_conf = [r for r in pool if r["combined_conf"] is not None]
    pool_sorted_conf = sorted(pool_with_conf, key=lambda r: r["combined_conf"], reverse=True)
    dataset_D = pool_sorted_conf[:N] if len(pool_with_conf) >= N else pool_sorted_conf

    # ── Eval set (shared) ─────────────────────────────────────────────────────
    eval_rows = [
        {"instruction": all_data[i]["instruction"],
         "input":       all_data[i]["input"],
         "output":      all_data[i]["output"]}
        for i in sorted(eval_indices)
    ]

    # ── Save ──────────────────────────────────────────────────────────────────
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving datasets ...")
    counts = {}
    counts["eval"]  = save_dataset(eval_rows, SPLITS_DIR / "eval.json", "Eval (shared)")
    counts["A"]     = save_dataset(dataset_A, DIRS["A"] / "train.json", "A  full")
    counts["B"]     = save_dataset(dataset_B, DIRS["B"] / "train.json", "B  random-20%")
    counts["C"]     = save_dataset(dataset_C, DIRS["C"] / "train.json", "C  IFD-top-20%")
    counts["D"]     = save_dataset(dataset_D, DIRS["D"] / "train.json", "D  Conf-top-20%")

    # ── Selection diagnostics ─────────────────────────────────────────────────
    import numpy as np

    def _stats(values, label):
        a = np.array(values, float)
        return {"label": label, "n": len(a),
                "min": round(float(np.min(a)),4), "mean": round(float(np.mean(a)),4),
                "max": round(float(np.max(a)),4), "std": round(float(np.std(a)),4)}

    ifd_C    = [r["ifd_score"]     for r in dataset_C]
    ifd_B    = [score_by_idx[r["index"]]["ifd_score"]
                for r in dataset_B
                if score_by_idx.get(r["index"], {}).get("ifd_score") is not None]
    conf_D   = [r["combined_conf"] for r in dataset_D if r["combined_conf"] is not None]
    conf_C   = [r["combined_conf"] for r in dataset_C if r["combined_conf"] is not None]

    print(f"\n{'='*60}")
    print(f"Dataset Summary")
    print(f"{'='*60}")
    print(f"  {'Group':<6}  {'Samples':>8}  Description")
    print(f"  {'─'*50}")
    print(f"  {'A':<6}  {counts['A']:>8}  All train (100%)")
    print(f"  {'B':<6}  {counts['B']:>8}  Random 20% of train")
    print(f"  {'C':<6}  {counts['C']:>8}  IFD top-20% (IFD≤1 pool)")
    print(f"  {'D':<6}  {counts['D']:>8}  Combined-conf top-20% (IFD≤1 pool)")
    print(f"  {'eval':<6}  {counts['eval']:>8}  Held-out eval (shared)")

    print(f"\n  IFD≤1 pool size      : {len(pool)}")
    print(f"  Pool coverage (C/D)  : {len(pool)}/{len(train_indices)} = {len(pool)/len(train_indices)*100:.1f}% of train")

    print(f"\n{'─'*60}")
    print(f"[IFD score: C vs B]")
    if ifd_C:
        s = _stats(ifd_C, "C (IFD-selected)")
        print(f"  C  min={s['min']:.4f}  mean={s['mean']:.4f}  max={s['max']:.4f}  std={s['std']:.4f}")
    if ifd_B:
        s = _stats(ifd_B, "B (random)")
        print(f"  B  min={s['min']:.4f}  mean={s['mean']:.4f}  max={s['max']:.4f}  std={s['std']:.4f}")

    print(f"\n[Combined confidence: D vs C]")
    if conf_D:
        s = _stats(conf_D, "D (conf-selected)")
        print(f"  D  min={s['min']:.4f}  mean={s['mean']:.4f}  max={s['max']:.4f}  std={s['std']:.4f}")
    if conf_C:
        s = _stats(conf_C, "C (IFD-selected)")
        print(f"  C  min={s['min']:.4f}  mean={s['mean']:.4f}  max={s['max']:.4f}  std={s['std']:.4f}")

    # ── Save split_info ───────────────────────────────────────────────────────
    info = {
        "seed": SEED,
        "num_eval": NUM_EVAL,
        "sel_ratio": SEL_RATIO,
        "N": N,
        "counts": counts,
        "pool_size": len(pool),
        "ifd_C_range": [round(min(ifd_C),4), round(max(ifd_C),4)] if ifd_C else None,
        "conf_D_range": [round(min(conf_D),4), round(max(conf_D),4)] if conf_D else None,
    }
    info_path = SPLITS_DIR / "split_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"\nSplit info saved → {info_path}")


if __name__ == "__main__":
    main()
