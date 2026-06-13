from typing import Dict, List, Tuple

from datasets import load_dataset

from src.data.gsm8k_noise import extract_final_answer


def load_gsm8k(
    num_train: int = 5000,
    num_val: int = 500,
    num_test: int = 500,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load GSM8K and return (train, val, test) as plain list-of-dicts.

    Source splits:
      train/val → HuggingFace 'train' split (7,473 samples total)
      test      → HuggingFace 'test'  split (1,319 samples total)

    Each sample:
      question       : str  — the math problem
      answer         : str  — full chain-of-thought ending with "#### N"
      correct_answer : str  — extracted final number (no commas)
    """
    ds_train = load_dataset("openai/gsm8k", "main", split="train")
    ds_test  = load_dataset("openai/gsm8k", "main", split="test")

    ds_train = ds_train.shuffle(seed=seed)
    ds_test  = ds_test.shuffle(seed=seed)

    def _cvt(row: Dict) -> Dict:
        return {
            "question":       row["question"],
            "answer":         row["answer"],
            "correct_answer": extract_final_answer(row["answer"]) or "",
        }

    n_tv = min(num_train + num_val, len(ds_train))
    train_val = [_cvt(r) for r in ds_train.select(range(n_tv))]

    n_te = min(num_test, len(ds_test))
    test = [_cvt(r) for r in ds_test.select(range(n_te))]

    return train_val[:num_train], train_val[num_train: num_train + num_val], test
