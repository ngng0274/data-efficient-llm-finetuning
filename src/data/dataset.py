from typing import Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# ── Prompt template ──────────────────────────────────────────────────────────
_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

_TEMPLATE_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def format_instruction(instruction: str, input_text: str = "") -> str:
    """Return the prompt prefix (everything up to and including '### Response:\\n')."""
    if input_text.strip():
        return _TEMPLATE_WITH_INPUT.format(
            instruction=instruction.strip(), input=input_text.strip()
        )
    return _TEMPLATE_NO_INPUT.format(instruction=instruction.strip())


# ── Dataset ──────────────────────────────────────────────────────────────────

class InstructionDataset(Dataset):
    """
    Tokenised instruction-tuning dataset with response-only loss masking.

    Labels are set to -100 for the instruction prefix so that cross-entropy
    is computed only on the response tokens (standard SFT practice).
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        max_length: int = 512,
        verbose: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded: List[Dict] = []
        self.metadata: List[Dict] = []

        for s in tqdm(samples, desc="Tokenising", disable=not verbose):
            instr_text = format_instruction(s["instruction"], s.get("input", ""))
            full_text = instr_text + s["output"] + tokenizer.eos_token

            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
            # Tokenise instruction prefix WITH special tokens (same flags as full_enc)
            # so the BOS/EOS handling is identical and instr_len matches the boundary
            # inside full_enc regardless of model family (LLaMA adds BOS; Qwen does not).
            instr_enc = tokenizer(
                instr_text,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )

            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]

            instr_len = min(len(instr_enc["input_ids"]), len(input_ids))

            labels = [-100] * instr_len + input_ids[instr_len:]

            self.encoded.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
            self.metadata.append(
                {
                    "is_clean": s.get("is_clean", True),
                    "noise_type": s.get("noise_type", "none"),
                }
            )

    # ── Dataset API ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> Dict:
        return self.encoded[idx]

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def n_clean(self) -> int:
        return sum(m["is_clean"] for m in self.metadata)

    @property
    def n_noisy(self) -> int:
        return len(self.metadata) - self.n_clean
