import json
import os
import re
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from src.data.gsm8k_dataset import format_gsm8k_prompt  # single source of truth

# Matches "#### 18" or "#### 1,234" anywhere in text (last occurrence wins)
_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*)")
# Fallback: any number (int or decimal) in text — we take the last one.
# Using r"-?\d+\.?\d*" so "18.0" is treated as one token, not ["18","0"].
_ANY_NUM_RE = re.compile(r"-?\d+\.?\d*")


def extract_predicted_answer(text: str) -> Optional[str]:
    """Extract the final answer from model-generated text.

    Priority:
      1. Last '#### N' pattern  (standard GSM8K format)
      2. Last standalone integer (fallback when model omits ####)
    Returns a clean string with no commas, or None if nothing found.
    """
    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        return matches[-1].group(1).replace(",", "")

    # Fallback: return the last number found anywhere in the generated text
    all_nums = _ANY_NUM_RE.findall(text)
    return all_nums[-1] if all_nums else None


def _normalize(s: Optional[str]) -> Optional[str]:
    """Normalise an answer string for exact-match comparison.

    Handles: commas, trailing .0, leading zeros.
    """
    if s is None:
        return None
    s = s.replace(",", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else s
    except ValueError:
        return s



def compute_gsm8k_accuracy(
    model,
    tokenizer,
    samples: List[Dict],
    max_new_tokens: int = 512,
    batch_size: int = 4,
    output_dir: Optional[str] = None,
    split_name: str = "test",
) -> Dict:
    """Generate solutions for `samples` and compute exact-match accuracy.

    Args:
        samples:       list of dicts with 'question' and 'correct_answer' keys
        max_new_tokens: generation budget (512 is enough for GSM8K CoT)
        output_dir:    if given, saves two JSON files there
        split_name:    prefix for saved file names (e.g. 'test', 'val')

    Returns:
        {
            "split":        str,
            "accuracy":     float,   # 0–100
            "n_correct":    int,
            "n_total":      int,
            "n_no_answer":  int,     # model produced no parseable number
        }

    Saved files (when output_dir is set):
        {split_name}_predictions.json  — per-sample details
        {split_name}_accuracy.json     — summary dict (same as return value)
    """
    device = next(model.parameters()).device
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"   # required for batched generation
    model.eval()

    records: List[Dict] = []

    for i in tqdm(range(0, len(samples), batch_size), desc=f"Evaluating [{split_name}]"):
        batch = samples[i: i + batch_size]
        prompts = [format_gsm8k_prompt(s["question"]) for s in batch]

        enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = enc["input_ids"].shape[1]
        for j, s in enumerate(batch):
            gen_tokens = out_ids[j][prompt_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            pred = extract_predicted_answer(gen_text)
            ref = s["correct_answer"]
            is_correct = _normalize(pred) == _normalize(ref)

            records.append({
                "question":         s["question"],
                "correct_answer":   ref,
                "predicted_answer": pred,
                "is_correct":       is_correct,
                "generated_text":   gen_text,
            })

    tokenizer.padding_side = original_padding_side

    n_total     = len(records)
    n_correct   = sum(r["is_correct"] for r in records)
    n_no_answer = sum(r["predicted_answer"] is None for r in records)
    accuracy    = n_correct / n_total * 100 if n_total > 0 else 0.0

    summary = {
        "split":       split_name,
        "accuracy":    round(accuracy, 2),
        "n_correct":   n_correct,
        "n_total":     n_total,
        "n_no_answer": n_no_answer,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pred_path = os.path.join(output_dir, f"{split_name}_predictions.json")
        acc_path  = os.path.join(output_dir, f"{split_name}_accuracy.json")
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        with open(acc_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  Predictions → {pred_path}")
        print(f"  Accuracy    → {acc_path}")

    return summary
