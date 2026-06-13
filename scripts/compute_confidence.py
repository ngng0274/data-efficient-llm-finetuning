"""
Compute SemiPEP-inspired confidence scores for GSM8K train samples.

Per-token CE losses from a single conditioned forward pass
(format_gsm8k_prompt(question) + answer):

  Global   = exp(-mean_ce)                      — overall generation ease
  Local    = exp(-mean(top-25% hard token CE))  — ease of hardest token chunk
  Combined = sqrt(Global × Local)               — geometric mean

Also loads ifd_scores.json (if available) and reports:
  - Distribution stats for each score (vs IFD std baseline of ~0.10)
  - Spearman ρ between each confidence measure and IFD
    (low ρ with IFD = independent signal = good for combining later)

Usage:
    python scripts/compute_confidence.py
    python scripts/compute_confidence.py --ifd_path outputs/gsm8k_ifd/ifd_scores.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gsm8k_dataset import format_gsm8k_prompt
from src.data.gsm8k_loader import load_gsm8k


# ── Core computation ──────────────────────────────────────────────────────────

def compute_per_token_losses(
    model, tokenizer, question: str, answer: str, max_length: int, device: str
):
    """
    Single conditioned forward pass.
    Returns np.ndarray of per-token CE losses for response tokens,
    or None if the response span is empty / truncated away.
    """
    whole_text = format_gsm8k_prompt(question) + answer

    input_ids = tokenizer.encode(
        whole_text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    total_tokens = input_ids.shape[1]

    # Locate response start in token space (same approach as compute_ifd.py)
    char_start = whole_text.rfind(answer)
    if char_start == -1:
        return None

    prefix_ids = tokenizer.encode(whole_text[:char_start])
    response_start = len(prefix_ids)

    if response_start >= total_tokens:
        return None

    labels = input_ids.clone()
    labels[0, :response_start] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    # Vectorised per-token loss: logits[pos-1] predicts token at pos
    logits = outputs.logits                                                  # [1, T, V]
    logits_slice = logits[0, response_start - 1 : total_tokens - 1]        # [n_resp, V]
    log_probs    = F.log_softmax(logits_slice, dim=-1)                      # [n_resp, V]
    true_tokens  = input_ids[0, response_start:total_tokens]                # [n_resp]
    losses = -log_probs.gather(1, true_tokens.unsqueeze(1)).squeeze(1)      # [n_resp]

    return losses.cpu().float().numpy()


def confidence_from_losses(losses: np.ndarray, hard_pct: float = 0.25):
    """
    Compute (global_conf, local_conf, combined_conf) from per-token CE losses.

    hard_pct : fraction of hardest tokens (by CE) used for Local confidence.
               Default 0.25 = top 25% highest-loss tokens.
    """
    # Global: geometric mean of per-token correct-token probabilities
    G = float(np.exp(-np.mean(losses)))

    # Local: geometric mean for the hardest (hard_pct * 100)% of tokens
    threshold   = np.percentile(losses, (1 - hard_pct) * 100)  # 75th pct for top-25%
    hard_losses = losses[losses >= threshold]
    if len(hard_losses) == 0:
        hard_losses = losses
    L = float(np.exp(-np.mean(hard_losses)))

    # Combined: geometric mean of G and L (mirrors SemiPEP joint confidence)
    C = float(np.sqrt(G * L))
    return G, L, C


# ── Reporting helpers ─────────────────────────────────────────────────────────

def print_dist(label: str, values: list, indent: int = 4):
    a = np.array(values)
    pad = " " * indent
    print(f"{pad}{label}  (n={len(a)})")
    print(f"{pad}  min    = {np.min(a):.4f}")
    print(f"{pad}  p25    = {np.percentile(a, 25):.4f}")
    print(f"{pad}  median = {np.median(a):.4f}")
    print(f"{pad}  mean   = {np.mean(a):.4f}")
    print(f"{pad}  p75    = {np.percentile(a, 75):.4f}")
    print(f"{pad}  max    = {np.max(a):.4f}")
    print(f"{pad}  std    = {np.std(a):.4f}")


def spearman(x, y):
    """Returns (rho, p_value)."""
    from scipy import stats
    return stats.spearmanr(x, y)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",        default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--num_train",  type=int, default=6000)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--save_path",  default="outputs/gsm8k_confidence/confidence_scores.json")
    parser.add_argument("--ifd_path",   default="outputs/gsm8k_ifd/ifd_scores.json",
                        help="Path to ifd_scores.json for correlation analysis")
    parser.add_argument("--checkpoint_every", type=int, default=200)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--hard_pct",   type=float, default=0.25,
                        help="Fraction of hardest tokens for Local confidence (default=0.25)")
    args = parser.parse_args()

    save_path      = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_path.parent / "_checkpoint.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Model      : {args.model_name}")
    print(f"Max length : {args.max_length}")
    print(f"Hard pct   : top {int(args.hard_pct * 100)}% hardest tokens → Local conf")

    # ── Load IFD scores for later correlation ─────────────────────────────────
    ifd_by_index: dict[int, float] = {}
    ifd_path = Path(args.ifd_path)
    if ifd_path.exists():
        with open(ifd_path, encoding="utf-8") as f:
            ifd_raw = json.load(f)
        for r in ifd_raw:
            if r.get("ifd_score") is not None:
                ifd_by_index[r["index"]] = r["ifd_score"]
        print(f"\nIFD scores loaded : {len(ifd_by_index)} valid samples from {args.ifd_path}")
    else:
        print(f"\nIFD file not found ({args.ifd_path}) — correlation analysis will be skipped")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model in bfloat16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if device == "cuda" else "cpu",
    )
    model.eval()
    model.config.use_cache = False

    if device == "cuda":
        print(f"  VRAM after load : {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading GSM8K train ({args.num_train} samples, seed={args.seed}) ...")
    train_data, _, _ = load_gsm8k(
        num_train=args.num_train, num_val=0, num_test=0, seed=args.seed
    )
    print(f"  Loaded {len(train_data)} samples")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    results = []
    start_idx = 0
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming from checkpoint : {start_idx}/{len(train_data)} done")

    # ── Confidence computation ────────────────────────────────────────────────
    print(f"\nComputing confidence [{start_idx} → {len(train_data)}] ...")
    start_time = time.time()

    for i in tqdm(range(start_idx, len(train_data)), desc="Confidence"):
        sample       = train_data[i]
        question     = sample["question"]
        answer       = sample["answer"]
        correct_ans  = sample.get("correct_answer", "")

        record = {
            "index":         i,
            "question":      question,
            "answer":        answer,
            "correct_answer": correct_ans,
            "global_conf":   None,
            "local_conf":    None,
            "combined_conf": None,
            "ifd_score":     ifd_by_index.get(i),   # None if IFD file absent
            "skipped":       False,
            "skip_reason":   None,
        }

        try:
            losses = compute_per_token_losses(
                model, tokenizer, question, answer, args.max_length, device
            )

            if losses is None or len(losses) == 0:
                record["skipped"]     = True
                record["skip_reason"] = "empty_response_span"
                results.append(record)
                continue

            G, L, C = confidence_from_losses(losses, hard_pct=args.hard_pct)
            record["global_conf"]   = G
            record["local_conf"]    = L
            record["combined_conf"] = C

        except Exception as exc:
            record["skipped"]     = True
            record["skip_reason"] = f"error: {exc}"

        results.append(record)

        if (i + 1) % args.checkpoint_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)

    # ── Final save ────────────────────────────────────────────────────────────
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # ── Distribution statistics ───────────────────────────────────────────────
    elapsed = (time.time() - start_time) / 60
    valid   = [r for r in results if r["global_conf"] is not None]
    skipped = [r for r in results if r["skipped"]]

    print(f"\n{'='*60}")
    print(f"Confidence Computation Complete  ({elapsed:.1f} min)")
    print(f"{'='*60}")
    print(f"Total   : {len(results)}")
    print(f"Valid   : {len(valid)}")
    print(f"Skipped : {len(skipped)}")
    if skipped:
        reasons: dict[str, int] = {}
        for r in skipped:
            k = r.get("skip_reason") or "unknown"
            reasons[k] = reasons.get(k, 0) + 1
        print(f"  Skip reasons : {reasons}")

    if not valid:
        print("No valid samples — exiting.")
        return

    print(f"\n{'─'*60}")
    print(f"[Distribution Statistics]")
    print_dist("Global   (exp(-mean_ce))",        [r["global_conf"]   for r in valid])
    print_dist("Local    (exp(-top25%_hard_ce))",  [r["local_conf"]    for r in valid])
    print_dist("Combined (sqrt(G×L))",             [r["combined_conf"] for r in valid])

    # ── Spearman correlation with IFD ─────────────────────────────────────────
    # Set A: IFD ≤ 1 only (cherry pool — the subset used for selection)
    paired_cherry = [
        r for r in valid
        if r["ifd_score"] is not None and r["ifd_score"] <= 1.0
    ]
    # Set B: all samples where IFD is available (broader view)
    paired_all = [r for r in valid if r["ifd_score"] is not None]

    if not paired_all:
        print(f"\n[Spearman Correlation] — skipped (no IFD scores matched)")
        print(f"Saved → {save_path}")
        return

    def _corr_block(label, paired):
        if len(paired) < 10:
            print(f"  (too few samples: {len(paired)})")
            return
        ifd = [r["ifd_score"]     for r in paired]
        G_  = [r["global_conf"]   for r in paired]
        L_  = [r["local_conf"]    for r in paired]
        C_  = [r["combined_conf"] for r in paired]

        rho_G, p_G = spearman(ifd, G_)
        rho_L, p_L = spearman(ifd, L_)
        rho_C, p_C = spearman(ifd, C_)

        print(f"\n  {label}  (n={len(paired)})")
        print(f"  {'Metric':<10}  {'ρ':>7}  {'p-value':>10}  Signal")
        print(f"  {'─'*50}")
        for name, rho, p in [("Global",   rho_G, p_G),
                              ("Local",    rho_L, p_L),
                              ("Combined", rho_C, p_C)]:
            if abs(rho) < 0.3:
                tag = "독립적 (차별화)"
            elif abs(rho) < 0.6:
                tag = "약한 상관 (부분 독립)"
            else:
                tag = "강한 상관 (유사 신호)"
            print(f"  {name:<10}  {rho:>+7.4f}  {p:>10.2e}  {tag}")

    print(f"\n{'─'*60}")
    print(f"[Spearman Rank Correlation: IFD vs Confidence]")
    _corr_block("IFD ≤ 1 only (cherry pool)", paired_cherry)
    if len(paired_all) > len(paired_cherry):
        _corr_block("All IFD (including > 1)",  paired_all)

    # ── IFD distribution reminder ─────────────────────────────────────────────
    if paired_all:
        ifd_vals = [r["ifd_score"] for r in paired_all]
        ifd_arr  = np.array(ifd_vals)
        print(f"\n{'─'*60}")
        print(f"[IFD Distribution (reference, for std comparison)]")
        print(f"    n      = {len(ifd_arr)}")
        print(f"    mean   = {ifd_arr.mean():.4f}")
        print(f"    std    = {ifd_arr.std():.4f}   ← IFD baseline")
        ifd_leq1 = ifd_arr[ifd_arr <= 1]
        if len(ifd_leq1) < len(ifd_arr):
            print(f"    (IFD ≤ 1: n={len(ifd_leq1)}, "
                  f"mean={ifd_leq1.mean():.4f}, std={ifd_leq1.std():.4f})")

    print(f"\nSaved → {save_path}")


if __name__ == "__main__":
    main()
