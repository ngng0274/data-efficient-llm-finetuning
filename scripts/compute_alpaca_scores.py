"""
Compute IFD + confidence scores for all 52K Alpaca samples in one pass.

Two forward passes per sample (conditioned + standalone):
  conditioned pass  →  IFD numerator  +  confidence (Global / Local / Combined)
  standalone pass   →  IFD denominator

IFD formula (Cherry LLM V2, verbatim):
  IFD = mean_ce(response | instruction) / mean_ce(response alone)
  IFD > 1  → filtered at selection time (kept in file for analysis)
  Cherry selection = HIGHEST IFD samples (≤ 1)

Confidence (SemiPEP-inspired heuristic):
  Global   = exp(-mean_ce_cond)
  Local    = exp(-mean(top-25%% hardest token ce_cond))
  Combined = sqrt(Global × Local)

Prompt format: identical to Cherry LLM reference (data_analysis.py PROMPT_DICT).
max_length = 512 (matches Cherry LLM's original --max_length 512).

Output: outputs/alpaca_scores/alpaca_scores.json
  Fields per record: index, has_input, ifd_score, loss_standalone,
                     loss_conditioned, global_conf, local_conf, combined_conf,
                     skipped, skip_reason
  (instruction/input/output omitted — join by index with alpaca_data.json)

Usage:
    python scripts/compute_alpaca_scores.py
    python scripts/compute_alpaca_scores.py --max_length 512 --checkpoint_every 200
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

from src.data.alpaca_loader import (
    RESPONSE_MARKER,
    format_alpaca_prompt,
    load_alpaca_all,
)


# ── Core: per-token CE losses from one forward pass ──────────────────────────

def response_token_losses(
    model,
    tokenizer,
    full_text: str,
    response_text: str,
    max_length: int,
    device: str,
):
    """
    Vectorised per-token CE losses for `response_text` tokens within `full_text`.

    Returns np.ndarray of shape [n_response_tokens], or None if the response
    span is empty or entirely truncated.
    """
    input_ids = tokenizer.encode(
        full_text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    total = input_ids.shape[1]

    char_start = full_text.rfind(response_text)
    if char_start == -1:
        return None

    resp_start = len(tokenizer.encode(full_text[:char_start]))
    if resp_start >= total:
        return None

    labels = input_ids.clone()
    labels[0, :resp_start] = -100

    with torch.no_grad():
        out = model(input_ids, labels=labels)

    # logits[pos-1] predicts token at pos
    logits_sl  = out.logits[0, resp_start - 1 : total - 1]           # [n, V]
    log_probs  = F.log_softmax(logits_sl, dim=-1)                     # [n, V]
    true_toks  = input_ids[0, resp_start:total]                       # [n]
    losses     = -log_probs.gather(1, true_toks.unsqueeze(1)).squeeze(1)  # [n]

    return losses.cpu().float().numpy()


# ── IFD helpers (Cherry LLM data_by_IFD.py logic) ────────────────────────────

def compute_ifd(losses_alone, losses_cond):
    """IFD = mean_ce_cond / mean_ce_alone.  Returns (ifd, mean_alone, mean_cond)."""
    m_alone = float(np.mean(losses_alone))
    m_cond  = float(np.mean(losses_cond))
    if m_alone == 0.0:
        return None, m_alone, m_cond
    return m_cond / m_alone, m_alone, m_cond


# ── Confidence helpers (SemiPEP-inspired) ─────────────────────────────────────

def compute_confidence(losses_cond, hard_pct: float = 0.25):
    """Global / Local / Combined from conditioned token losses."""
    G = float(np.exp(-np.mean(losses_cond)))
    thr = np.percentile(losses_cond, (1 - hard_pct) * 100)   # 75th pct → top 25%
    hard = losses_cond[losses_cond >= thr]
    if len(hard) == 0:
        hard = losses_cond
    L = float(np.exp(-np.mean(hard)))
    C = float(np.sqrt(G * L))
    return G, L, C


# ── Reporting helpers ─────────────────────────────────────────────────────────

def dist_stats(values, label, indent=2):
    a = np.array(values, dtype=float)
    pad = " " * indent
    print(f"{pad}{label}  (n={len(a)})")
    for name, val in [
        ("min   ", np.min(a)),
        ("p25   ", np.percentile(a, 25)),
        ("median", np.median(a)),
        ("mean  ", np.mean(a)),
        ("p75   ", np.percentile(a, 75)),
        ("max   ", np.max(a)),
        ("std   ", np.std(a)),
    ]:
        print(f"{pad}  {name} = {val:.4f}")


def spearman_rho(x, y):
    """Spearman rank correlation (numpy-only, no scipy required)."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    # t-statistic → two-tailed p-value approximation (accurate for large n)
    t = rho * np.sqrt((n - 2) / max(1 - rho ** 2, 1e-15))
    # Use complementary error function as normal approximation (valid n > 30)
    from math import erfc, sqrt
    p = float(erfc(abs(t) / sqrt(2)))
    return rho, p


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",       default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--data_path",        default=None,
                        help="Path to alpaca_data.json (default: Cherry LLM reference copy)")
    parser.add_argument("--max_length",       type=int, default=512,
                        help="Token limit — matches Cherry LLM original (512)")
    parser.add_argument("--hard_pct",         type=float, default=0.25,
                        help="Top fraction of hardest tokens for Local confidence")
    parser.add_argument("--save_dir",         default="outputs/alpaca_scores")
    parser.add_argument("--checkpoint_every", type=int, default=200)
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path       = save_dir / "alpaca_scores.json"
    checkpoint_path = save_dir / "_checkpoint.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Model      : {args.model_name}")
    print(f"Max length : {args.max_length}  (Cherry LLM default=512)")
    print(f"Hard pct   : top {int(args.hard_pct*100)}%% hardest tokens → Local conf")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.pad_token = tok.eos_token

    print(f"Loading model in bfloat16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if device == "cuda" else "cpu",
    )
    model.eval()
    model.config.use_cache = False

    if device == "cuda":
        print(f"  VRAM after load : {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # RESPONSE_MARKER token count (for standalone max_length budget)
    marker_tok_len = len(tok.encode(RESPONSE_MARKER))

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading Alpaca (all 52K, original order) ...")
    data = load_alpaca_all(args.data_path)
    print(f"  Loaded {len(data)} samples")

    # ── Resume ────────────────────────────────────────────────────────────────
    results = []
    start_idx = 0
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming from checkpoint : {start_idx}/{len(data)} done")

    # ── Scoring loop ──────────────────────────────────────────────────────────
    print(f"\nScoring [{start_idx} → {len(data)}] ...")
    t0 = time.time()

    for i in tqdm(range(start_idx, len(data)), desc="Scoring"):
        s = data[i]
        instruction = s["instruction"]
        inp         = s["input"]
        output      = s["output"]
        has_input   = s["has_input"]

        rec = dict(
            index         = i,
            has_input     = has_input,
            ifd_score     = None,
            loss_standalone  = None,
            loss_conditioned = None,
            global_conf   = None,
            local_conf    = None,
            combined_conf = None,
            skipped       = False,
            skip_reason   = None,
        )

        if not output.strip():
            rec["skipped"]     = True
            rec["skip_reason"] = "empty_output"
            results.append(rec)
            continue

        try:
            prompt     = format_alpaca_prompt(instruction, inp)
            whole_text = prompt + output          # conditioned
            direct_text = RESPONSE_MARKER + output  # standalone

            # ── Prompt token length (sets answer budget for standalone) ───────
            prompt_tok_len = len(
                tok.encode(prompt, truncation=True, max_length=args.max_length)
            )
            if prompt_tok_len >= args.max_length:
                rec["skipped"]     = True
                rec["skip_reason"] = "prompt_too_long"
                results.append(rec)
                continue

            # standalone budget = marker tokens + same answer budget as conditioned
            standalone_max = marker_tok_len + (args.max_length - prompt_tok_len)

            # ── Forward passes ────────────────────────────────────────────────
            losses_alone = response_token_losses(
                model, tok, direct_text, output, standalone_max, device
            )
            losses_cond  = response_token_losses(
                model, tok, whole_text,  output, args.max_length, device
            )

            if losses_alone is None or losses_cond is None:
                rec["skipped"]     = True
                rec["skip_reason"] = "empty_response_span"
                results.append(rec)
                continue

            n_alone = len(losses_alone)
            n_cond  = len(losses_cond)

            if n_alone <= 0 or n_cond <= 0:
                rec["skipped"]     = True
                rec["skip_reason"] = "zero_length_response"
                results.append(rec)
                continue

            # Cherry LLM data_by_IFD.py line 100 check
            if prompt_tok_len + n_alone > args.max_length:
                rec["skipped"]     = True
                rec["skip_reason"] = "response_overflow"
                results.append(rec)
                continue

            # ── IFD ──────────────────────────────────────────────────────────
            ifd, m_alone, m_cond = compute_ifd(losses_alone, losses_cond)
            if ifd is None:
                rec["skipped"]     = True
                rec["skip_reason"] = "zero_standalone_loss"
                results.append(rec)
                continue

            rec["ifd_score"]        = float(ifd)
            rec["loss_standalone"]  = float(m_alone)
            rec["loss_conditioned"] = float(m_cond)

            # ── Confidence (reuses conditioned losses) ────────────────────────
            G, L, C = compute_confidence(losses_cond, args.hard_pct)
            rec["global_conf"]   = G
            rec["local_conf"]    = L
            rec["combined_conf"] = C

        except Exception as exc:
            rec["skipped"]     = True
            rec["skip_reason"] = f"error: {exc}"

        results.append(rec)

        if (i + 1) % args.checkpoint_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)   # compact (no indent) to keep size down

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    elapsed = (time.time() - t0) / 60

    # ── Statistics ────────────────────────────────────────────────────────────
    valid   = [r for r in results if r["ifd_score"] is not None]
    skipped = [r for r in results if r["skipped"]]
    gt1     = [r for r in valid   if r["ifd_score"] > 1.0]
    pool    = [r for r in valid   if r["ifd_score"] <= 1.0]   # cherry candidate pool

    print(f"\n{'='*60}")
    print(f"Alpaca Scoring Complete  ({elapsed:.1f} min)")
    print(f"{'='*60}")
    print(f"Total         : {len(results)}")
    print(f"Valid         : {len(valid)}")
    print(f"  IFD ≤ 1     : {len(pool)}   (cherry candidate pool)")
    print(f"  IFD > 1     : {len(gt1)}    (will be filtered at selection)")
    print(f"Skipped       : {len(skipped)}")
    if skipped:
        reasons: dict = {}
        for r in skipped:
            k = r.get("skip_reason") or "unknown"
            reasons[k] = reasons.get(k, 0) + 1
        print(f"  Skip reasons: {reasons}")

    sep = f"{'─'*60}"

    # ── IFD distribution ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"[IFD Distribution]")
    if valid:
        dist_stats([r["ifd_score"] for r in valid], "All valid (incl. IFD > 1)")
    if pool:
        dist_stats([r["ifd_score"] for r in pool],  "IFD ≤ 1 pool (cherry candidates)")

    # ── Confidence distributions ──────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"[Confidence Distributions]  (computed on conditioned pass)")
    if valid:
        dist_stats([r["global_conf"]   for r in valid], "Global   (exp(-mean_ce))")
        dist_stats([r["local_conf"]    for r in valid], "Local    (exp(-top25%_hard_ce))")
        dist_stats([r["combined_conf"] for r in valid], "Combined (sqrt(G×L))")

    # ── Spearman correlations ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"[Spearman ρ : IFD vs Confidence]")

    def _corr_block(label, subset):
        if len(subset) < 10:
            print(f"  {label}: too few samples ({len(subset)})")
            return
        ifd_v = [r["ifd_score"]     for r in subset]
        g_v   = [r["global_conf"]   for r in subset]
        l_v   = [r["local_conf"]    for r in subset]
        c_v   = [r["combined_conf"] for r in subset]
        rho_g, p_g = spearman_rho(ifd_v, g_v)
        rho_l, p_l = spearman_rho(ifd_v, l_v)
        rho_c, p_c = spearman_rho(ifd_v, c_v)
        print(f"\n  {label}  (n={len(subset)})")
        print(f"  {'Metric':<10}  {'ρ':>7}  {'p-value':>10}  Signal")
        print(f"  {'─'*52}")
        for name, rho, p in [("Global",   rho_g, p_g),
                              ("Local",    rho_l, p_l),
                              ("Combined", rho_c, p_c)]:
            if abs(rho) < 0.3:
                tag = "독립 (차별화)"
            elif abs(rho) < 0.5:
                tag = "약한 상관 (부분 독립)"
            elif abs(rho) < 0.7:
                tag = "중간 상관"
            else:
                tag = "강한 상관 (유사 신호)"
            print(f"  {name:<10}  {rho:>+7.4f}  {p:>10.2e}  {tag}")

    _corr_block("IFD ≤ 1 only (cherry pool)",   pool)
    _corr_block("All valid   (incl. IFD > 1)",  valid)

    # ── Key questions summary ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"[Key Questions]")
    if pool:
        ifd_std = np.std([r["ifd_score"] for r in pool])
        print(f"  IFD std (pool ≤1) = {ifd_std:.4f}  "
              f"{'> GSM8K 0.10 ✓' if ifd_std > 0.10 else '≤ GSM8K 0.10 ✗'}")
    if valid:
        ifd_std_all = np.std([r["ifd_score"] for r in valid])
        loc_std  = np.std([r["local_conf"]  for r in valid])
        glob_std = np.std([r["global_conf"] for r in valid])
        print(f"  IFD std (all)     = {ifd_std_all:.4f}")
        print(f"  Local conf std    = {loc_std:.4f}  "
              f"{'> IFD std ✓' if loc_std > ifd_std_all else '≤ IFD std ✗'}")
        print(f"  Global conf std   = {glob_std:.4f}")

    print(f"\nSaved → {save_path}")


if __name__ == "__main__":
    main()
