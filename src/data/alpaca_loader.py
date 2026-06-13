"""
Alpaca data loader — prompt format identical to Cherry LLM reference.

Prompt source: reference/Cherry_LLM/cherry_seletion/data_analysis.py PROMPT_DICT

  with input:
    "Below is an instruction that describes a task, paired with an input
     that provides further context. Write a response that appropriately
     completes the request.\n\n
     ### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"

  without input:
    "Below is an instruction that describes a task. Write a response that
     appropriately completes the request.\n\n
     ### Instruction:\n{instruction}\n\n### Response:"

Full training text  : prompt + output
Response marker     : "### Response:"  (no trailing newline — output appended directly)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Default path relative to project root
_DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "reference" / "Cherry_LLM" / "data" / "alpaca_data.json"

# Cherry LLM prompt templates (verbatim)
PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

# Marker at the end of every prompt — used by IFD/confidence scripts to
# locate where the response (output) begins in the full text.
RESPONSE_MARKER = "### Response:"


def format_alpaca_prompt(instruction: str, input_text: str = "") -> str:
    """Return the prompt string ending with '### Response:'.

    Append output directly to get the full training / scoring text:
        full_text = format_alpaca_prompt(instruction, input) + output
    """
    if input_text.strip():
        return PROMPT_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_NO_INPUT.format(instruction=instruction)


def load_alpaca(
    data_path: str = None,
    num_eval: int = 1000,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Load Alpaca JSON and return (train_samples, eval_samples).

    Each sample dict contains:
        instruction : str
        input       : str   (empty string if none)
        output      : str
        prompt      : str   (pre-formatted, ends with '### Response:')
        has_input   : bool

    Args:
        data_path : path to alpaca_data.json  (default: Cherry LLM reference copy)
        num_eval  : samples held out for training-time loss monitoring
        seed      : shuffle seed (keep consistent across all scripts)

    Returns:
        (train_samples, eval_samples)
        — eval is NOT used for IFD/confidence scoring, only for training val loss.
        — IFD/confidence scripts should call load_alpaca_all() for scoring.
    """
    path = Path(data_path) if data_path else _DEFAULT_DATA_PATH
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    samples = [_convert(r) for r in raw]

    rng = random.Random(seed)
    rng.shuffle(samples)

    return samples[num_eval:], samples[:num_eval]   # (train, eval)


def load_alpaca_all(data_path: str = None) -> List[Dict]:
    """Return all 52K samples in original order (for IFD / confidence scoring).

    Index in this list == 'index' field used by scoring scripts.
    No shuffle so that index is stable across runs.
    """
    path = Path(data_path) if data_path else _DEFAULT_DATA_PATH
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [_convert(r) for r in raw]


def _convert(r: Dict) -> Dict:
    instruction = r["instruction"]
    input_text  = r.get("input", "")
    output      = r["output"]
    return {
        "instruction": instruction,
        "input":       input_text,
        "output":      output,
        "prompt":      format_alpaca_prompt(instruction, input_text),
        "has_input":   bool(input_text.strip()),
    }


# ── Quick verification (python src/data/alpaca_loader.py) ────────────────────
if __name__ == "__main__":
    import textwrap

    print("Loading Alpaca data ...")
    train, eval_ = load_alpaca(num_eval=1000, seed=42)
    all_data = load_alpaca_all()

    print(f"\n{'='*65}")
    print(f"Dataset stats")
    print(f"{'='*65}")
    print(f"  Total (all)  : {len(all_data)}")
    print(f"  Train split  : {len(train)}")
    print(f"  Eval  split  : {len(eval_)}")
    with_input = sum(1 for s in all_data if s["has_input"])
    print(f"  Has input    : {with_input} ({with_input/len(all_data)*100:.1f}%)")
    print(f"  No input     : {len(all_data)-with_input} ({(len(all_data)-with_input)/len(all_data)*100:.1f}%)")

    # Show 3 representative samples: 2 without input, 1 with input
    no_inp  = [s for s in all_data if not s["has_input"]]
    has_inp = [s for s in all_data if s["has_input"]]
    samples_to_show = [no_inp[0], no_inp[1], has_inp[0]]

    for idx, s in enumerate(samples_to_show, 1):
        print(f"\n{'─'*65}")
        print(f"Sample {idx}  (has_input={s['has_input']})")
        print(f"{'─'*65}")
        print(f"[PROMPT]\n{s['prompt']}")
        # Show first 200 chars of output
        output_preview = s["output"][:200].replace("\n", "\\n")
        if len(s["output"]) > 200:
            output_preview += "..."
        print(f"\n[OUTPUT (preview)]\n{output_preview}")
        print(f"\n[FULL TEXT (first 300 chars)]\n"
              f"{(s['prompt'] + s['output'])[:300].replace(chr(10), chr(10)+'  ')}")
