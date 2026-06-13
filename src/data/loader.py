import random
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset


def load_alpaca(num_samples: int = 10000, seed: int = 42) -> List[Dict]:
    """Download tatsu-lab/alpaca and return as plain list of dicts."""
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    if num_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    return [dict(row) for row in dataset]


def split_dataset(
    samples: List[Dict],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split into train / val / test.

    Count takes priority over ratio when both are given.
    Pass n_val / n_test as integers for exact counts (e.g. smoke tests).
    """
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    n = len(samples)
    _n_test = n_test if n_test is not None else int(n * test_ratio)
    _n_val  = n_val  if n_val  is not None else int(n * val_ratio)

    assert _n_test + _n_val < n, (
        f"val({_n_val}) + test({_n_test}) >= total({n}). "
        "Reduce n_val / n_test or increase num_samples."
    )

    test_idx  = indices[:_n_test]
    val_idx   = indices[_n_test : _n_test + _n_val]
    train_idx = indices[_n_test + _n_val :]

    return (
        [samples[i] for i in train_idx],
        [samples[i] for i in val_idx],
        [samples[i] for i in test_idx],
    )
