import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

MODEL_PATH = BASE_DIR / "iris_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

# Friendly names accepted by the API
API_FIELDS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def load_resources():
    if not MODEL_PATH.is_file() or not METRICS_PATH.is_file():
        raise FileNotFoundError("Trained model or metrics not found.")

    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)

    return model, metrics


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "iris-classification-api"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required"}), 400

    missing = [field for field in API_FIELDS if field not in data]
    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    try:
        values = [float(data[field]) for field in API_FIELDS]
    except (ValueError, TypeError):
        return jsonify({
            "error": "All flower measurements must be numeric"
        }), 400

    if not all(pd.notna(value) for value in values):
        return jsonify({"error": "Measurements must be finite numbers"}), 400

    try:
        model, metrics = load_resources()
        sample = pd.DataFrame(
            [values],
            columns=metrics["features"]
        )
        prediction = str(model.predict(sample)[0])

        return jsonify({
            "prediction": prediction,
            "features": dict(zip(API_FIELDS, values))
        })
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
