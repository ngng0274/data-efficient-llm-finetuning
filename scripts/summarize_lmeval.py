"""lm-eval 결과 JSON을 읽어 비교 표를 출력한다."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "eval_results"

MODELS = [
    ("Baseline",   "baseline"),
    ("A: Full",    "alpaca_A_full"),
    ("B: Random",  "alpaca_B_random"),
    ("C: IFD",     "alpaca_C_ifd"),
    ("D: Conf",    "alpaca_D_confidence"),
]

TASKS = {
    "arc_challenge": "ARC-C",
    "hellaswag":     "HellaSwag",
}


def find_result_json(model_dir: Path) -> Path | None:
    """results.json 또는 results_<datetime>.json을 재귀 탐색."""
    if not model_dir.exists():
        return None
    matches = sorted(model_dir.glob("**/results*.json"), reverse=True)
    return matches[0] if matches else None


def extract_acc(data: dict, task_key: str) -> str:
    results = data.get("results", {})
    # lm-eval v0.4+ 키 형식: "arc_challenge" 또는 "arc_challenge,none=0"
    for k, v in results.items():
        if task_key in k:
            acc = v.get("acc_norm,none") or v.get("acc_norm") or v.get("acc,none") or v.get("acc")
            if acc is not None:
                return f"{acc * 100:.2f}"
    return "N/A"


rows = []
for label, name in MODELS:
    model_dir = RESULTS_DIR / name
    json_path = find_result_json(model_dir)
    if json_path is None:
        rows.append((label, "—", "—"))
        continue
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    arc  = extract_acc(data, "arc_challenge")
    hswag = extract_acc(data, "hellaswag")
    rows.append((label, arc, hswag))

# 표 출력
header = f"{'Model':<14} {'ARC-C (0-shot)':>16} {'HellaSwag (0-shot)':>20}"
sep    = "-" * len(header)
print(sep)
print(header)
print(sep)
for label, arc, hswag in rows:
    print(f"{label:<14} {arc:>16} {hswag:>20}")
print(sep)
print("\n* acc_norm basis (normalized accuracy)")
print("* N/A : evaluation not complete or result file missing")
