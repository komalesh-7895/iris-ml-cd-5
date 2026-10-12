import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent


def train_model():
    data = pd.read_csv(BASE_DIR / "Iris.csv")
    data.columns = data.columns.str.strip()

    target = next(
        (c for c in data.columns if c.lower() == "species"),
        None
    )
    if target is None:
        raise ValueError("Iris.csv must contain a Species column.")

    X = data.drop(columns=[target]).select_dtypes(include="number")
    X = X.loc[
        :, ~X.columns.str.lower().isin(["id", "unnamed: 0"])
    ]

    if X.shape[1] != 4:
        raise ValueError(
            f"Expected 4 numeric features; found {X.shape[1]}."
        )

    if X.isna().any().any():
        raise ValueError("Feature columns contain missing values.")

    y = data[target].astype(str).str.strip()

    if y.nunique() != 3:
        raise ValueError("Expected exactly 3 Iris species classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "features": list(X.columns),
        "classes": sorted(y.unique().tolist()),
        "classification_report": classification_report(
            y_test, predictions,
            output_dict=True,
            zero_division=0
        )
    }

    joblib.dump(model, BASE_DIR / "iris_model.pkl")

    with open(BASE_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    data.to_csv(BASE_DIR / "iris_dataset.csv", index=False)

    print(f"Accuracy: {accuracy:.4f}")
    print("Created iris_model.pkl, metrics.json, iris_dataset.csv")


if __name__ == "__main__":
    train_model()
