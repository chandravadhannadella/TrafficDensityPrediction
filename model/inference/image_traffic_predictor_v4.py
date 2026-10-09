"""ImageTrafficPredictorV4: Image-specific traffic density prediction model."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model.inference.model_config import NoOpScaler, resolve_model_paths
from model.vehicle_detector import detect_vehicles

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_DIR = Path(__file__).resolve().parents[1] / "saved_models"
DEFAULT_METADATA_PATH = MODEL_DIR / "model_metadata_v4_image.json"
DEFAULT_PROXY_LABEL_NOTE = (
    "V4-IMAGE traffic-density labels are proxy labels derived from total_vehicle_count thresholds "
    "(LOW <= 8, MEDIUM 9-15, HIGH >= 16) and are not verified human ground-truth labels."
)


class ImageTrafficPredictorV4:
    """Dedicated image-level traffic density prediction using V4-image model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        feature_names_path: str | Path | None = None,
    ) -> None:
        if model_path is None and scaler_path is None and feature_names_path is None:
            model_path, scaler_path, feature_names_path = resolve_model_paths("v4_image")

        self.model_path = Path(model_path) if model_path is not None else MODEL_DIR / "traffic_density_model_v4_image.pkl"
        self.scaler_path = Path(scaler_path) if scaler_path is not None else MODEL_DIR / "scaler_v4_image.pkl"
        self.feature_names_path = Path(feature_names_path) if feature_names_path is not None else MODEL_DIR / "feature_names_v4_image.json"

        print(f"MODEL VERSION: V4-IMAGE (Image-Specific)")
        self.model = self._load_model(self.model_path)
        loaded_scaler = self._load_pickle(self.scaler_path)
        self.scaler = loaded_scaler if loaded_scaler is not None else NoOpScaler()
        self.feature_names = self._load_feature_names(self.feature_names_path)
        self.label_order = ["LOW", "MEDIUM", "HIGH"]

    @staticmethod
    def _load_pickle(path: Path):
        with path.open("rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def _load_model(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path}")
        return ImageTrafficPredictorV4._load_pickle(path)

    @staticmethod
    def _load_feature_names(path: Path) -> list[str]:
        if not path.exists():
            raise FileNotFoundError(f"Missing feature metadata: {path}")
        with path.open("r", encoding="utf-8") as handle:
            feature_names = json.load(handle)
        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError(f"Feature metadata in {path} is empty or invalid.")
        return list(feature_names)

    @staticmethod
    def _load_metadata(path: Path | None = None) -> dict[str, Any]:
        metadata_path = path or DEFAULT_METADATA_PATH
        if not metadata_path.exists():
            return {}
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _proxy_label_note() -> str:
        metadata = ImageTrafficPredictorV4._load_metadata(DEFAULT_METADATA_PATH)
        return str(metadata.get("proxy_label_note") or DEFAULT_PROXY_LABEL_NOTE)

    @staticmethod
    def _valid_image_path(image_path: str | Path) -> Path:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {path.suffix}. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}")
        return path

    def _prepare_feature_row(self, counts: dict[str, int], total: int) -> pd.DataFrame:
        """Prepare image-level features for the V4-image model."""
        row = {
            "car_count": float(counts.get("car", 0)),
            "bus_count": float(counts.get("bus", 0)),
            "truck_count": float(counts.get("truck", 0)),
            "motorcycle_count": float(counts.get("motorcycle", 0)),
            "bicycle_count": float(counts.get("bicycle", 0)),
            "total_vehicle_count": float(total),
        }
        missing = [name for name in self.feature_names if name not in row]
        if missing:
            raise ValueError(
                f"Image model requires features {missing} that are not provided. "
                f"Available: {list(row.keys())}"
            )
        prepared = pd.DataFrame([row], columns=self.feature_names)
        if prepared.shape[1] != len(self.feature_names):
            raise ValueError("Image feature count does not match the saved model feature list.")
        return prepared

    def predict_image(self, image_path: str | Path) -> dict[str, Any]:
        """Predict traffic density for a single image."""
        path = self._valid_image_path(image_path)
        processed_image, counts, total = detect_vehicles(path, model_name="yolov8n.pt", confidence=0.25, image_size=640)
        del processed_image

        feature_frame = self._prepare_feature_row(counts, total)
        scaled = self.scaler.transform(feature_frame)
        probabilities = self.model.predict_proba(scaled)[0]
        predicted_label = str(self.model.classes_[int(np.argmax(probabilities))])
        confidence = float(np.max(probabilities))
        class_to_prob = {label: float(probabilities[index]) for index, label in enumerate(self.model.classes_)}

        return {
            "image": path.name,
            "overall_traffic_density": predicted_label,
            "overall_confidence": round(confidence, 4),
            "total_vehicles": int(total),
            "counts": {key: int(counts.get(key, 0)) for key in ["car", "bus", "truck", "motorcycle", "bicycle"]},
            "probabilities": {key: round(float(class_to_prob.get(key, 0.0)), 4) for key in self.label_order},
            "model_version": "v4_image",
            "model_info": "V4-IMAGE: dedicated image-specific model trained on 924 raw images with proxy labels",
            "proxy_label_note": self._proxy_label_note(),
        }
