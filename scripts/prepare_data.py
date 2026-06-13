"""
Prepare and inspect the noisy training data.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config configs/stage1_config.yaml --out data/prepared
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_alpaca, split_dataset
from src.data.noise_injector import NoiseInjector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_config.yaml")
    parser.add_argument("--out", default="data/prepared")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Download ──────────────────────────────────────────────────────────────
    print(f"Downloading Alpaca ({data_cfg['num_samples']} samples) ...")
    samples = load_alpaca(num_samples=data_cfg["num_samples"], seed=data_cfg["seed"])
    print(f"  Downloaded {len(samples)} samples.")

    # ── Split ─────────────────────────────────────────────────────────────────
    train_raw, val_raw, test_raw = split_dataset(
        samples,
        val_ratio=data_cfg.get("val_ratio", 0.1),
        test_ratio=data_cfg.get("test_ratio", 0.1),
        n_val=data_cfg.get("n_val"),
        n_test=data_cfg.get("n_test"),
        seed=data_cfg["seed"],
    )
    print(f"  Split → train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}")

    # ── Inject noise into train ───────────────────────────────────────────────
    injector = NoiseInjector(
        noise_ratio=data_cfg["noise_ratio"],
        noise_types=data_cfg["noise_types"],
        seed=data_cfg["seed"],
    )
    train_noisy = injector.inject(train_raw)
    stats = injector.stats(train_noisy)

    print(f"\nNoise injection stats:")
    print(f"  Total  : {stats['total']}")
    print(f"  Clean  : {stats['clean']}  ({stats['clean']/stats['total']*100:.1f}%)")
    print(f"  Noisy  : {stats['noisy']}  ({stats['noise_ratio']*100:.1f}%)")
    for t, cnt in stats["by_type"].items():
        print(f"    {t:<25}: {cnt}")

    # ── Save ─────────────────────────────────────────────────────────────────
    splits = {
        "train_noisy": train_noisy,
        "val": val_raw,
        "test": test_raw,
    }
    for name, data in splits.items():
        path = out_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(data):>5} samples → {path}")

    # ── Print noisy examples for inspection ──────────────────────────────────
    noisy_examples = [s for s in train_noisy if not s["is_clean"]][:3]
    print("\n── 3 noisy examples ──────────────────────────────────────────")
    for i, ex in enumerate(noisy_examples, 1):
        print(f"\n[{i}] noise_type = {ex['noise_type']}")
        print(f"  instruction : {ex['instruction'][:80]}...")
        print(f"  output      : {ex['output'][:120]}...")

    examples_path = out_dir / "noisy_examples.json"
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(noisy_examples, f, indent=2, ensure_ascii=False)
    print(f"\nSaved noisy examples → {examples_path}")


if __name__ == "__main__":
    main()
