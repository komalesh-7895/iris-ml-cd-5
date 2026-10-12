import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


class TestIrisModel(unittest.TestCase):

    def test_model_exists(self):
        self.assertTrue((BASE_DIR / "iris_model.pkl").is_file())

    def test_metrics_exist(self):
        self.assertTrue((BASE_DIR / "metrics.json").is_file())

    def test_model_accuracy(self):
        with open(BASE_DIR / "metrics.json", encoding="utf-8") as f:
            metrics = json.load(f)

        self.assertGreaterEqual(metrics["accuracy"], 0.90)

    def test_prediction_returns_valid_species(self):
        with open(BASE_DIR / "metrics.json", encoding="utf-8") as f:
            metrics = json.load(f)

        model = joblib.load(BASE_DIR / "iris_model.pkl")

        sample = pd.DataFrame(
            [[5.1, 3.5, 1.4, 0.2]],
            columns=metrics["features"]
        )

        prediction = str(model.predict(sample)[0])
        self.assertIn(prediction, metrics["classes"])


if __name__ == "__main__":
    unittest.main()
