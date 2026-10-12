import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

required = [
    "iris_model.pkl",
    "metrics.json",
    "iris_dataset.csv",
]

for name in required:
    if not (BASE_DIR / name).is_file():
        raise SystemExit(f"QUALITY GATE FAILED: Missing {name}")

with open(BASE_DIR / "metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)

if metrics["accuracy"] < 0.90:
    raise SystemExit("QUALITY GATE FAILED: Accuracy below 90%")

if not metrics.get("features") or len(metrics["features"]) != 4:
    raise SystemExit("QUALITY GATE FAILED: Invalid feature configuration")

print(f"QUALITY GATE PASSED: Accuracy = {metrics['accuracy']:.4f}")
