import random
import re
from typing import Dict, List, Optional

import numpy as np

# GSM8K final answer format: "#### 18"  or  "#### 1,234"
_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*)")

# Minimum change thresholds for wrong-answer generation.
# A candidate is accepted only if |candidate - correct| >= min_delta,
# where min_delta = max(MIN_ABSOLUTE, round(abs(correct) * MIN_RELATIVE)).
# Examples:
#   correct=4   → min_delta = max(2, round(0.8)) = 2   → 4±1 rejected
#   correct=50  → min_delta = max(2, round(10))  = 10  → 50±5 rejected
#   correct=200 → min_delta = max(2, round(40))  = 40  → 200±20 rejected
_MIN_ABSOLUTE = 2
_MIN_RELATIVE = 0.20


def extract_final_answer(answer_text: str) -> Optional[str]:
    """Return the final answer as a clean integer string (no commas), or None."""
    m = _ANSWER_RE.search(answer_text)
    return m.group(1).replace(",", "") if m else None


def _make_wrong_number(correct: int, rng: random.Random) -> int:
    """Return an integer that differs from `correct` by at least min_delta."""
    abs_val = abs(correct) if correct != 0 else 1
    min_delta = max(_MIN_ABSOLUTE, round(abs_val * _MIN_RELATIVE))

    candidates = []
    # Multiplicative — naturally large gap
    for factor in [2, 3, 4, 5]:
        candidates.append(correct * factor)
    for divisor in [2, 3]:
        candidates.append(correct // divisor)
    # Additive — delta drawn from [min_delta, min_delta*3+20] to guarantee gap
    hi = max(min_delta * 3, min_delta + 20)
    for _ in range(6):
        delta = rng.randint(min_delta, hi)
        candidates.append(correct + delta)
        candidates.append(correct - delta)

    # Keep only candidates with sufficient change
    valid = [c for c in candidates if abs(c - correct) >= min_delta]
    return rng.choice(valid) if valid else correct + rng.choice([-1, 1]) * min_delta


class GSM8KAnswerCorruptor:
    """
    Corrupts the final '#### N' answer in GSM8K samples while keeping
    the chain-of-thought intact.  This directly poisons the learning signal
    (correct reasoning → wrong label), making noise effects clearly measurable.

    Each sample gains:
      is_noisy   : bool   — ground truth flag for later filtering evaluation
      is_clean   : bool   — inverse of is_noisy (kept for code compatibility)
      noise_type : str    — "answer_corruption" | "none"
      wrong_answer: str   — the injected wrong number (noisy samples only)
    """

    NOISE_TYPE = "answer_corruption"

    def __init__(self, noise_ratio: float = 0.50, seed: int = 42):
        assert 0.0 <= noise_ratio <= 1.0
        self.noise_ratio = noise_ratio
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def inject(self, samples: List[Dict]) -> List[Dict]:
        n = len(samples)
        n_noisy = int(n * self.noise_ratio)
        noisy_idx = set(self.np_rng.choice(n, size=n_noisy, replace=False).tolist())

        result = []
        for i, sample in enumerate(samples):
            s = dict(sample)
            if i in noisy_idx:
                s = self._corrupt(s)
                s["is_noisy"] = True
                s["is_clean"] = False
                s["noise_type"] = self.NOISE_TYPE
            else:
                s["is_noisy"] = False
                s["is_clean"] = True
                s["noise_type"] = "none"
            result.append(s)
        return result

    def stats(self, samples: List[Dict]) -> Dict:
        total = len(samples)
        noisy = sum(1 for s in samples if s.get("is_noisy", False))
        return {
            "total": total,
            "clean": total - noisy,
            "noisy": noisy,
            "noise_ratio": noisy / total if total else 0.0,
        }

    def _corrupt(self, sample: Dict) -> Dict:
        m = _ANSWER_RE.search(sample["answer"])
        if m is None:
            return sample

        try:
            correct_int = int(m.group(1).replace(",", ""))
        except ValueError:
            return sample

        wrong_int = _make_wrong_number(correct_int, self.rng)
        sample["answer"] = sample["answer"][: m.start()] + f"#### {wrong_int}"
        sample["wrong_answer"] = str(wrong_int)
        return sample
