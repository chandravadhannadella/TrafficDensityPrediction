from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model" / "saved_models"

DEFAULT_MODEL_VERSION = os.getenv("TRAFFIC_MODEL_VERSION", "v2").strip().lower()


class NoOpScaler:
    """Compatibility scaler used when a model does not require preprocessing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)

    def fit_transform(self, X, y=None):
        return np.asarray(X)

    def inverse_transform(self, X):
        return np.asarray(X)


def resolve_model_paths(model_version: str | None = None) -> tuple[Path, Path, Path]:
    version = (model_version or DEFAULT_MODEL_VERSION).strip().lower()
    if version == "v3":
        return (
            MODEL_DIR / "traffic_density_model_v3.pkl",
            MODEL_DIR / "scaler_v3.pkl",
            MODEL_DIR / "feature_names_v3.json",
        )
    if version == "v4_image":
        return (
            MODEL_DIR / "traffic_density_model_v4_image.pkl",
            MODEL_DIR / "scaler_v4_image.pkl",
            MODEL_DIR / "feature_names_v4_image.json",
        )
    if version in {"v2", "v2_default"}:
        return (
            MODEL_DIR / "traffic_density_model_v2.pkl",
            MODEL_DIR / "scaler_v2.pkl",
            MODEL_DIR / "feature_names_v2.json",
        )
    return (
        MODEL_DIR / "traffic_density_model.pkl",
        MODEL_DIR / "scaler.pkl",
        MODEL_DIR / "feature_names.json",
    )
