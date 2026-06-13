"""
Verify GSM8K data loading and noise injection.
Prints 5 noisy examples to confirm the pipeline works.

Usage:
    python scripts/verify_gsm8k_noise.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gsm8k_loader import load_gsm8k
from src.data.gsm8k_noise import GSM8KAnswerCorruptor


def main():
    print("Loading GSM8K (train=50, val=10, test=10) ...")
    train, val, test = load_gsm8k(num_train=50, num_val=10, num_test=10, seed=42)
    print(f"  train={len(train)}, val={len(val)}, test={len(test)}")

    corruptor = GSM8KAnswerCorruptor(noise_ratio=0.50, seed=42)
    train_noisy = corruptor.inject(train)

    stats = corruptor.stats(train_noisy)
    print(f"  clean={stats['clean']}, noisy={stats['noisy']} ({stats['noise_ratio']*100:.0f}% noise)\n")

    noisy_samples = [s for s in train_noisy if s["is_noisy"]][:5]

    for i, s in enumerate(noisy_samples, 1):
        print("=" * 65)
        print(f"[Example {i}]")
        q = s["question"]
        print(f"Question : {q[:120]}{'...' if len(q) > 120 else ''}")
        print(f"Correct  : {s['correct_answer']}")
        print(f"Wrong    : {s['wrong_answer']}")
        # Show the last 2 lines of the (corrupted) answer
        tail = "\n".join(s["answer"].splitlines()[-3:])
        print(f"Answer tail:\n  {tail}")
        print(f"is_noisy={s['is_noisy']}  noise_type={s['noise_type']}")

    print("=" * 65)
    print("\n[Clean example for comparison]")
    clean_sample = next(s for s in train_noisy if not s["is_noisy"])
    print(f"Question : {clean_sample['question'][:120]}")
    print(f"Correct  : {clean_sample['correct_answer']}")
    tail = "\n".join(clean_sample["answer"].splitlines()[-3:])
    print(f"Answer tail:\n  {tail}")
    print(f"is_noisy={clean_sample['is_noisy']}  noise_type={clean_sample['noise_type']}")


if __name__ == "__main__":
    main()
