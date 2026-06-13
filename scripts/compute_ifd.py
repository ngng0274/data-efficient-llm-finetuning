"""
Compute IFD (Instruction-Following Difficulty) scores for GSM8K train data.
Cherry LLM V2 approach: base model directly, no pre-experience stage.

IFD = mean_token_loss(response | instruction) / mean_token_loss(response alone)
  - IFD close to 0 : instruction greatly reduces generation difficulty
  - IFD close to 1 : instruction provides little benefit (still <= 1)
  - IFD > 1        : instruction hurts; filtered out at selection time

Cherry selection uses HIGHEST IFD samples (IFD <= 1).

Output: outputs/gsm8k_ifd/ifd_scores.json
  Every record is kept (including skipped), with ifd_score=null for invalid ones.
  The selection script (select_by_ifd.py) does the actual filtering.

Usage:
    python scripts/compute_ifd.py
    python scripts/compute_ifd.py --model_name Qwen/Qwen2.5-1.5B --num_train 6000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gsm8k_dataset import format_gsm8k_prompt
from src.data.gsm8k_loader import load_gsm8k

log_softmax = nn.LogSoftmax(dim=-1)

# Must match the tail of format_gsm8k_prompt exactly
RESPONSE_MARKER = "### Solution:\n"


def get_response_token_losses(
    model, tokenizer, full_text: str, response_text: str, max_length: int, device: str
):
    """
    Return per-token cross-entropy losses for response_text tokens within full_text.

    Args:
        full_text     : complete text fed to the model (prefix + response)
        response_text : the response portion whose loss we measure
        max_length    : token truncation limit for full_text

    Returns:
        (n_tokens, loss_array)  — n_tokens==0 means the response was truncated away.
    """
    input_ids = tokenizer.encode(
        full_text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    total_tokens = input_ids.shape[1]

    # Locate where the response starts in character space, then map to token space.
    # rfind is used (same as Cherry LLM original) to handle cases where the response
    # string might also appear in the prefix.
    char_start = full_text.rfind(response_text)
    if char_start == -1:
        return 0, None

    prefix_token_ids = tokenizer.encode(full_text[:char_start])
    response_start = len(prefix_token_ids)

    if response_start >= total_tokens:
        return 0, None

    n_response = total_tokens - response_start

    # Forward pass with labels masked for prefix tokens
    labels = input_ids.clone()
    labels[0, :response_start] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    # Vectorised per-token loss extraction (avoids Python loop over positions)
    logits = outputs.logits  # [1, total_tokens, vocab_size]
    # logits[pos-1] predicts token at position pos
    logits_slice = logits[0, response_start - 1 : total_tokens - 1]  # [n_response, vocab]
    log_probs = log_softmax(logits_slice)                              # [n_response, vocab]
    true_tokens = input_ids[0, response_start:total_tokens]           # [n_response]
    losses = -log_probs.gather(1, true_tokens.unsqueeze(1)).squeeze(1)  # [n_response]

    return n_response, losses.cpu().float().numpy()


def main():
    parser = argparse.ArgumentParser(description="Compute IFD scores for GSM8K train data")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--num_train", type=int, default=6000,
                        help="Number of train samples to score (must match training split)")
    parser.add_argument("--max_length", type=int, default=768,
                        help="Token length limit — match training max_length")
    parser.add_argument("--save_path", default="outputs/gsm8k_ifd/ifd_scores.json")
    parser.add_argument("--checkpoint_every", type=int, default=200,
                        help="Flush checkpoint to disk every N samples (resume support)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Data shuffle seed — must match training loader seed")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_path.parent / "_checkpoint.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Model      : {args.model_name}")
    print(f"Max length : {args.max_length}")
    print(f"Seed       : {args.seed}")

    # ── Load model (base model, inference only, no LoRA) ─────────────────────
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
    model.config.use_cache = False  # not needed for single forward pass

    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM after load : {allocated:.2f} GB")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading GSM8K train ({args.num_train} samples) ...")
    train_data, _, _ = load_gsm8k(
        num_train=args.num_train, num_val=0, num_test=0, seed=args.seed
    )
    print(f"  Loaded {len(train_data)} samples")

    # RESPONSE_MARKER token length — used to set standalone answer budget
    marker_token_len = len(tokenizer.encode(RESPONSE_MARKER))

    # ── Resume from checkpoint if interrupted ─────────────────────────────────
    results = []
    start_idx = 0
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming from checkpoint : {start_idx}/{len(train_data)} done")

    # ── IFD computation ───────────────────────────────────────────────────────
    print(f"\nComputing IFD scores [{start_idx} → {len(train_data)}] ...")
    start_time = time.time()

    for i in tqdm(range(start_idx, len(train_data)), desc="IFD"):
        sample = train_data[i]
        question = sample["question"]
        answer = sample["answer"]
        correct_answer = sample.get("correct_answer", "")

        record = {
            "index": i,
            "question": question,
            "answer": answer,
            "correct_answer": correct_answer,
            "ifd_score": None,
            "loss_standalone": None,
            "loss_conditioned": None,
            "skipped": False,
            "skip_reason": None,
        }

        try:
            whole_text = format_gsm8k_prompt(question) + answer   # conditioned
            direct_text = RESPONSE_MARKER + answer                 # standalone

            # Token length of prompt — sets the answer budget for standalone text
            prompt_token_len = len(
                tokenizer.encode(
                    format_gsm8k_prompt(question),
                    truncation=True,
                    max_length=args.max_length,
                )
            )

            if prompt_token_len >= args.max_length:
                record["skipped"] = True
                record["skip_reason"] = "prompt_too_long"
                results.append(record)
                continue

            # standalone max_length: marker tokens + same answer budget as conditioned
            standalone_max = marker_token_len + (args.max_length - prompt_token_len)

            n_alone, losses_alone = get_response_token_losses(
                model, tokenizer, direct_text, answer, standalone_max, device
            )
            n_cond, losses_cond = get_response_token_losses(
                model, tokenizer, whole_text, answer, args.max_length, device
            )

            if n_alone <= 0 or n_cond <= 0:
                record["skipped"] = True
                record["skip_reason"] = "empty_response_span"
                results.append(record)
                continue

            # Identical check from Cherry LLM data_by_IFD.py line 100
            if prompt_token_len + n_alone > args.max_length:
                record["skipped"] = True
                record["skip_reason"] = "response_overflow"
                results.append(record)
                continue

            mean_alone = float(np.mean(losses_alone))
            mean_cond = float(np.mean(losses_cond))

            if mean_alone == 0.0:
                record["skipped"] = True
                record["skip_reason"] = "zero_standalone_loss"
                results.append(record)
                continue

            # IFD = conditioned loss / standalone loss  (Cherry LLM formula)
            ifd = mean_cond / mean_alone

            record["ifd_score"] = ifd
            record["loss_standalone"] = mean_alone
            record["loss_conditioned"] = mean_cond
            # IFD > 1 samples are NOT skipped here — kept for analysis.
            # select_by_ifd.py will filter them out.

        except Exception as exc:
            record["skipped"] = True
            record["skip_reason"] = f"error: {exc}"

        results.append(record)

        # Checkpoint flush
        if (i + 1) % args.checkpoint_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)

    # ── Final save ────────────────────────────────────────────────────────────
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = (time.time() - start_time) / 60
    valid = [r for r in results if r["ifd_score"] is not None]
    cherry_pool = [r for r in valid if r["ifd_score"] <= 1.0]
    gt1 = [r for r in valid if r["ifd_score"] > 1.0]
    skipped = [r for r in results if r["skipped"]]

    print(f"\n{'='*55}")
    print(f"IFD Computation Complete  ({elapsed:.1f} min)")
    print(f"{'='*55}")
    print(f"Total                    : {len(results)}")
    print(f"Valid (IFD computed)     : {len(valid)}")
    print(f"  IFD <= 1 (cherry pool) : {len(cherry_pool)}")
    print(f"  IFD >  1 (filtered)    : {len(gt1)}")
    print(f"Skipped                  : {len(skipped)}")

    if cherry_pool:
        scores = [r["ifd_score"] for r in cherry_pool]
        print(f"\nIFD distribution (IFD <= 1 pool):")
        print(f"  min  = {min(scores):.4f}")
        print(f"  p25  = {float(np.percentile(scores, 25)):.4f}")
        print(f"  mean = {float(np.mean(scores)):.4f}")
        print(f"  p75  = {float(np.percentile(scores, 75)):.4f}")
        print(f"  max  = {max(scores):.4f}")

    print(f"\nSaved → {save_path}")

    # Skip reason breakdown
    if skipped:
        reasons: dict[str, int] = {}
        for r in skipped:
            key = r.get("skip_reason") or "unknown"
            reasons[key] = reasons.get(key, 0) + 1
        print(f"\nSkip reasons: {reasons}")


if __name__ == "__main__":
    main()
