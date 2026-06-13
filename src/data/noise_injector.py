import random
import re
from typing import Dict, List, Optional

import numpy as np


class NoiseInjector:
    """
    Injects three types of noise into Alpaca-style instruction-response samples.

    noise_types:
      response_swap      – replace output with another random sample's output
      truncated_response – keep only 10-40% of the output (word-level)
      shuffled_sentences – shuffle sentence order inside the output
    """

    SUPPORTED = ["response_swap", "truncated_response", "shuffled_sentences"]

    def __init__(
        self,
        noise_ratio: float = 0.40,
        noise_types: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.noise_ratio = noise_ratio
        self.noise_types = noise_types or self.SUPPORTED
        for t in self.noise_types:
            assert t in self.SUPPORTED, f"Unknown noise type: {t}"
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    # ── public ──────────────────────────────────────────────────────────────

    def inject(self, samples: List[Dict]) -> List[Dict]:
        """Return a new list with noise injected.  Each item gains
        ``is_clean: bool`` and ``noise_type: str`` fields."""
        n = len(samples)
        n_noisy = int(n * self.noise_ratio)
        noisy_indices = set(
            self.np_rng.choice(n, size=n_noisy, replace=False).tolist()
        )

        result = []
        for i, sample in enumerate(samples):
            s = dict(sample)
            if i in noisy_indices:
                noise_type = self.rng.choice(self.noise_types)
                s = self._apply(s, samples, noise_type, i)
                s["is_clean"] = False
                s["noise_type"] = noise_type
            else:
                s["is_clean"] = True
                s["noise_type"] = "none"
            result.append(s)
        return result

    def stats(self, samples: List[Dict]) -> Dict:
        total = len(samples)
        noisy = sum(1 for s in samples if not s.get("is_clean", True))
        by_type: Dict[str, int] = {}
        for s in samples:
            t = s.get("noise_type", "none")
            by_type[t] = by_type.get(t, 0) + 1
        return {
            "total": total,
            "clean": total - noisy,
            "noisy": noisy,
            "noise_ratio": noisy / total,
            "by_type": by_type,
        }

    # ── private ─────────────────────────────────────────────────────────────

    def _apply(self, sample: Dict, all_samples: List[Dict], noise_type: str, idx: int) -> Dict:
        if noise_type == "response_swap":
            return self._swap(sample, all_samples, idx)
        if noise_type == "truncated_response":
            return self._truncate(sample)
        if noise_type == "shuffled_sentences":
            return self._shuffle_sentences(sample)
        return sample

    def _swap(self, sample: Dict, all_samples: List[Dict], idx: int) -> Dict:
        candidates = [i for i in range(len(all_samples)) if i != idx]
        other = self.rng.choice(candidates)
        sample["output"] = all_samples[other]["output"]
        return sample

    def _truncate(self, sample: Dict) -> Dict:
        words = sample["output"].split()
        if len(words) <= 4:
            return sample
        keep = max(2, int(len(words) * self.rng.uniform(0.10, 0.40)))
        sample["output"] = " ".join(words[:keep]) + "..."
        return sample

    def _shuffle_sentences(self, sample: Dict) -> Dict:
        # Split on sentence-ending punctuation followed by a space
        sentences = re.split(r"(?<=[.!?])\s+", sample["output"].strip())
        sentences = [s for s in sentences if s]
        if len(sentences) <= 1:
            return sample
        self.rng.shuffle(sentences)
        sample["output"] = " ".join(sentences)
        return sample
