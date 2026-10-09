from __future__ import annotations

from pathlib import Path

import pandas as pd

from model.inference.image_traffic_predictor import ImageTrafficPredictor
from model.training.train_density_model import get_feature_columns

ROOT = Path(__file__).resolve().parents[1]


def test_get_feature_columns_excludes_target_and_proxy_columns() -> None:
    columns = [
        "avg_car_count",
        "avg_bus_count",
        "avg_total_vehicle_count",
        "max_vehicle_count",
        "unique_vehicle_count",
        "traffic_flow",
        "traffic_density",
        "congestion_score",
        "video_id",
        "window_start",
        "window_end",
    ]

    features = get_feature_columns(columns)

    assert features == ["avg_car_count", "avg_bus_count", "max_vehicle_count", "unique_vehicle_count", "traffic_flow"]
    assert "traffic_density" not in features
    assert "avg_total_vehicle_count" not in features
    assert "congestion_score" not in features


def test_final_split_contains_only_allowed_feature_columns() -> None:
    train_df = pd.read_csv(ROOT / "data" / "datasets" / "final" / "train.csv")

    features = get_feature_columns(train_df.columns)
    assert len(features) > 0
    assert "traffic_density" not in features
    assert "avg_total_vehicle_count" not in features
    assert "congestion_score" not in features


def test_v3_inference_accepts_saved_noop_scaler() -> None:
    predictor = ImageTrafficPredictor(
        model_path=ROOT / "model" / "saved_models" / "traffic_density_model_v3.pkl",
        scaler_path=ROOT / "model" / "saved_models" / "scaler_v3.pkl",
        feature_names_path=ROOT / "model" / "saved_models" / "feature_names_v3.json",
    )

    assert predictor.model is not None
    assert predictor.scaler is not None
    assert hasattr(predictor.scaler, "transform")
