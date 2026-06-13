from typing import Dict, List

import numpy as np
import torch
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq

from src.data.dataset import format_instruction


# ── Perplexity ───────────────────────────────────────────────────────────────

def compute_perplexity(
    model,
    dataset,
    tokenizer,
    batch_size: int = 4,
) -> float:
    """
    Average token-level cross-entropy loss on *response* tokens, exponentiated.
    Uses the -100 label mask already embedded in the dataset.
    """
    device = next(model.parameters()).device
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    total_nll, total_tokens = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Perplexity", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            # outputs.loss is the mean NLL over non-masked tokens in the batch.
            # We need to accumulate the *sum* to compute a corpus-level average.
            n_tokens = (batch["labels"] != -100).sum().item()
            if n_tokens == 0:
                continue
            total_nll += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_nll = total_nll / max(total_tokens, 1)
    return float(np.exp(avg_nll))


# ── ROUGE ────────────────────────────────────────────────────────────────────

def compute_rouge(
    model,
    tokenizer,
    samples: List[Dict],
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> Dict[str, float]:
    """
    Generate responses for ``samples`` and compute ROUGE-1/2/L against gold.

    Returns scores as percentages (0-100).
    """
    device = next(model.parameters()).device
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    predictions: List[str] = []
    references: List[str] = []

    # Generation requires left-padding so every prompt ends at the same position
    # and the model generates immediately after the last real token.
    # Training uses right-padding, so we save and restore the original setting.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    model.eval()
    for i in tqdm(range(0, len(samples), batch_size), desc="Generating"):
        batch_samples = samples[i : i + batch_size]
        prompts = [
            format_instruction(s["instruction"], s.get("input", ""))
            for s in batch_samples
        ]

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

        for j, out in enumerate(out_ids):
            prompt_len = enc["input_ids"].shape[1]
            gen_tokens = out[prompt_len:]
            pred = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            ref = batch_samples[j]["output"].strip()
            predictions.append(pred)
            references.append(ref)

    tokenizer.padding_side = original_padding_side   # restore

    agg: Dict[str, List[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in agg:
            agg[k].append(scores[k].fmeasure)

    return {k: round(float(np.mean(v)) * 100, 2) for k, v in agg.items()}
