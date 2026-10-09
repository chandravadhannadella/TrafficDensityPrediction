from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from model.prediction.traffic_predictor import TrafficDensityPredictor

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "saved_models"


def test_saved_model_predicts() -> None:
    test_df = pd.read_csv(ROOT / "data" / "datasets" / "final" / "test.csv")

    with (MODEL_DIR / "feature_names.json").open("r", encoding="utf-8") as handle:
        feature_names = json.load(handle)

    predictor = TrafficDensityPredictor()
    predictions = predictor.predict(test_df[feature_names])

    assert len(predictions) == len(test_df)
    assert all(label in {"LOW", "MEDIUM", "HIGH"} for label in predictions)


if __name__ == "__main__":
    test_saved_model_predicts()
    print("Saved model prediction check passed.")
