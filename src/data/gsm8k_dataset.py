from typing import Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def format_gsm8k_prompt(question: str) -> str:
    """Prompt template shared by training and evaluation.

    Must stay identical in both places — any divergence invalidates eval.
    """
    return (
        "Below is a math problem. Solve it step by step.\n"
        "At the end, write your final answer after '####'.\n\n"
        "### Problem:\n"
        f"{question.strip()}\n\n"
        "### Solution:\n"
    )


class GSM8KDataset(Dataset):
    """Tokenised GSM8K dataset with prompt-masked labels (loss on answer only).

    Each sample must have 'question' and 'answer' keys.
    Noisy samples have their 'answer' already corrupted by GSM8KAnswerCorruptor.
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        max_length: int = 768,
        verbose: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded: List[Dict] = []
        self.metadata: List[Dict] = []

        for s in tqdm(samples, desc="Tokenising", disable=not verbose):
            prompt = format_gsm8k_prompt(s["question"])
            full_text = prompt + s["answer"] + tokenizer.eos_token

            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
            # Tokenise prompt with the same flags to get the exact boundary index
            prompt_enc = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )

            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]
            prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))

            labels = [-100] * prompt_len + input_ids[prompt_len:]

            self.encoded.append(
                {
                    "input_ids":      input_ids,
                    "attention_mask": attention_mask,
                    "labels":         labels,
                }
            )
            self.metadata.append(
                {
                    "is_noisy":       s.get("is_noisy", False),
                    "noise_type":     s.get("noise_type", "none"),
                    "correct_answer": s.get("correct_answer", ""),
                }
            )

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> Dict:
        return self.encoded[idx]

    @property
    def n_clean(self) -> int:
        return sum(1 for m in self.metadata if not m["is_noisy"])

    @property
    def n_noisy(self) -> int:
        return sum(1 for m in self.metadata if m["is_noisy"])
