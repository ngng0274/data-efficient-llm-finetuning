"""
Compare epoch accuracy curves between noisy and clean experiments.

Usage:
    # Smoke test comparison
    python scripts/compare_gsm8k_results.py \
        --noisy outputs/gsm8k_smoke_noisy \
        --clean outputs/gsm8k_smoke_clean

    # Full experiment comparison
    python scripts/compare_gsm8k_results.py \
        --noisy outputs/gsm8k_stage1_noisy \
        --clean outputs/gsm8k_stage1_clean

    # Custom output path
    python scripts/compare_gsm8k_results.py \
        --noisy outputs/gsm8k_stage1_noisy \
        --clean outputs/gsm8k_stage1_clean \
        --output outputs/comparison.png
"""

import argparse
import json
import os
import sys


def load_epoch_accuracy(output_dir: str) -> dict:
    path = os.path.join(output_dir, "epoch_accuracy.json")
    if not os.path.exists(path):
        print(f"  [WARNING] epoch_accuracy.json not found: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_table(noisy_data: dict, clean_data: dict):
    noisy_map = {r["epoch"]: r for r in noisy_data.get("epochs", [])}
    clean_map = {r["epoch"]: r for r in clean_data.get("epochs", [])}
    all_epochs = sorted(set(noisy_map) | set(clean_map))

    noisy_ratio = noisy_data.get("noise_ratio", "?")
    print(f"\n{'Epoch':>6}  {'Noisy (%)':>10}  {'Clean (%)':>10}  {'Gap (C-N)':>10}")
    print("-" * 44)
    for e in all_epochs:
        n_acc = noisy_map[e]["accuracy"] if e in noisy_map else float("nan")
        c_acc = clean_map[e]["accuracy"] if e in clean_map else float("nan")
        gap   = c_acc - n_acc if (e in noisy_map and e in clean_map) else float("nan")
        n_str = f"{n_acc:.2f}" if n_acc == n_acc else "  —"
        c_str = f"{c_acc:.2f}" if c_acc == c_acc else "  —"
        g_str = f"{gap:+.2f}" if gap == gap else "  —"
        print(f"{e:>6}  {n_str:>10}  {c_str:>10}  {g_str:>10}")
    print()


def plot_comparison(noisy_data: dict, clean_data: dict, output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    noisy_epochs = [r["epoch"]    for r in noisy_data.get("epochs", [])]
    noisy_accs   = [r["accuracy"] for r in noisy_data.get("epochs", [])]
    clean_epochs = [r["epoch"]    for r in clean_data.get("epochs", [])]
    clean_accs   = [r["accuracy"] for r in clean_data.get("epochs", [])]

    noisy_ratio = noisy_data.get("noise_ratio", 0)
    noisy_label = f"Noisy ({int(noisy_ratio*100)}%)"
    clean_label = "Clean (100%)"

    fig, ax = plt.subplots(figsize=(9, 5))

    if noisy_epochs:
        ax.plot(noisy_epochs, noisy_accs, color="coral", linewidth=2,
                marker="o", markersize=7, label=noisy_label)
    if clean_epochs:
        ax.plot(clean_epochs, clean_accs, color="steelblue", linewidth=2,
                marker="s", markersize=7, label=clean_label)

    # Annotate final gap
    if noisy_accs and clean_accs:
        last_n = noisy_accs[-1]
        last_c = clean_accs[-1]
        last_e = max(noisy_epochs[-1], clean_epochs[-1])
        gap = last_c - last_n
        ax.annotate(
            f"gap = {gap:+.1f}%",
            xy=(last_e, (last_n + last_c) / 2),
            xytext=(last_e - 0.6, (last_n + last_c) / 2 + 1),
            fontsize=10, color="gray",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
        )

    all_epochs = sorted(set(noisy_epochs) | set(clean_epochs))
    ax.set_xticks(all_epochs)
    ax.set_title("Noisy vs Clean — Test Accuracy per Epoch (GSM8K)", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy",  required=True, help="Noisy experiment output dir")
    parser.add_argument("--clean",  required=True, help="Clean experiment output dir")
    parser.add_argument("--output", default=None,  help="Path for comparison PNG")
    args = parser.parse_args()

    print(f"Loading noisy: {args.noisy}")
    noisy_data = load_epoch_accuracy(args.noisy)
    print(f"Loading clean: {args.clean}")
    clean_data = load_epoch_accuracy(args.clean)

    if not noisy_data and not clean_data:
        print("No epoch_accuracy.json found in either directory. Exiting.")
        sys.exit(1)

    print_table(noisy_data, clean_data)

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.noisy)),
        "gsm8k_comparison.png",
    )
    plot_comparison(noisy_data, clean_data, output_path)


if __name__ == "__main__":
    main()
