from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model" / "saved_models"


class TrafficDensityPredictor:
    """Load the saved proxy-label model and predict traffic density for feature rows."""

    def __init__(self, model_path: str | Path | None = None, scaler_path: str | Path | None = None, feature_names_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else MODEL_DIR / "traffic_density_model.pkl"
        self.scaler_path = Path(scaler_path) if scaler_path is not None else MODEL_DIR / "scaler.pkl"
        self.feature_names_path = Path(feature_names_path) if feature_names_path is not None else MODEL_DIR / "feature_names.json"

        self.model = self._load_pickle(self.model_path)
        self.scaler = self._load_pickle(self.scaler_path)
        self.feature_names = self._load_feature_names(self.feature_names_path)

    @staticmethod
    def _load_pickle(path: Path):
        with path.open("rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def _load_feature_names(path: Path) -> list[str]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return list(data)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self.feature_names if name not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        prepared = df[self.feature_names].copy()
        return prepared

    def predict(self, df: pd.DataFrame) -> list[str]:
        prepared = self._prepare_features(df)
        scaled = self.scaler.transform(prepared)
        return self.model.predict(scaled).tolist()

    def predict_one(self, feature_row: dict[str, float]) -> str:
        frame = pd.DataFrame([feature_row])
        prediction = self.predict(frame)
        return prediction[0]
