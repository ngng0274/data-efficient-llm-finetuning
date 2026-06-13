"""
Tokenised Alpaca dataset — prompt masked to -100, loss on output only.

Compatible with DataCollatorForSeq2Seq (pads input_ids, attention_mask, labels).
"""

from typing import Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.alpaca_loader import format_alpaca_prompt


class AlpacaDataset(Dataset):
    """Tokenised Alpaca dataset.

    Each sample must have 'instruction', 'input' (may be empty), 'output' keys.
    Prompt tokens are masked (-100) so training loss is computed on output only.
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        max_length: int = 512,
        verbose: bool = True,
    ):
        self.max_length = max_length
        self.encoded: List[Dict] = []

        for s in tqdm(samples, desc="Tokenising", disable=not verbose):
            prompt    = format_alpaca_prompt(s["instruction"], s.get("input", ""))
            full_text = prompt + s["output"] + tokenizer.eos_token

            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
            prompt_enc = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )

            input_ids      = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]
            prompt_len     = min(len(prompt_enc["input_ids"]), len(input_ids))

            # Mask prompt tokens — loss only on output
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            self.encoded.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         labels,
            })

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> Dict:
        return self.encoded[idx]
