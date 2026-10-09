"""Regression tests for image vs video separation and V4-image model."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from model.inference.image_traffic_predictor import ImageTrafficPredictor
from model.inference.image_traffic_predictor_v4 import ImageTrafficPredictorV4
from model.inference.video_traffic_predictor import VideoTrafficPredictor

ROOT = Path(__file__).resolve().parents[1]


class TestImageModelSeparation:
    """Ensure image and video models are separate and don't interfere."""

    def test_image_predictor_v4_loads_v4_image_model(self) -> None:
        """ImageTrafficPredictor should load V4-image by default."""
        predictor = ImageTrafficPredictor()
        assert predictor.model_version == "v4_image"
        assert (ROOT / "model" / "saved_models" / "traffic_density_model_v4_image.pkl").exists()

    def test_video_predictor_still_uses_v3(self) -> None:
        """VideoTrafficPredictor should continue using V3 (unchanged)."""
        predictor = VideoTrafficPredictor()
        assert (ROOT / "model" / "saved_models" / "traffic_density_model_v3.pkl").exists()

    def test_image_v4_feature_names_correct(self) -> None:
        """V4-image uses simple image-level features."""
        predictor = ImageTrafficPredictor(model_version="v4_image")
        expected_features = {
            "car_count",
            "bus_count",
            "truck_count",
            "motorcycle_count",
            "bicycle_count",
            "total_vehicle_count",
        }
        assert set(predictor.feature_names) == expected_features

    def test_image_v3_feature_names_still_available(self) -> None:
        """V3 (legacy) features should still be loadable if explicitly requested."""
        model_path = ROOT / "model" / "saved_models" / "traffic_density_model_v3.pkl"
        if model_path.exists():
            predictor = ImageTrafficPredictor(model_version="v3")
            expected_features = {
                "avg_car_count",
                "avg_bus_count",
                "avg_truck_count",
                "avg_motorcycle_count",
                "avg_bicycle_count",
                "max_vehicle_count",
                "unique_vehicle_count",
                "traffic_flow",
            }
            assert set(predictor.feature_names) == expected_features


class TestV4ImageModelArtifacts:
    """Verify all V4-image model artifacts exist and are valid."""

    def test_v4_image_model_file_exists(self) -> None:
        """V4-image model pickle file should exist."""
        model_file = ROOT / "model" / "saved_models" / "traffic_density_model_v4_image.pkl"
        assert model_file.exists(), f"Model file not found: {model_file}"

    def test_v4_image_feature_names_file_exists(self) -> None:
        """V4-image feature names JSON should exist."""
        feature_file = ROOT / "model" / "saved_models" / "feature_names_v4_image.json"
        assert feature_file.exists(), f"Feature names file not found: {feature_file}"

    def test_v4_image_scaler_file_exists(self) -> None:
        """V4-image scaler pickle should exist."""
        scaler_file = ROOT / "model" / "saved_models" / "scaler_v4_image.pkl"
        assert scaler_file.exists(), f"Scaler file not found: {scaler_file}"

    def test_v4_image_metadata_file_exists(self) -> None:
        """V4-image metadata JSON should exist."""
        metadata_file = ROOT / "model" / "saved_models" / "model_metadata_v4_image.json"
        assert metadata_file.exists(), f"Metadata file not found: {metadata_file}"

    def test_v4_image_model_loads_without_error(self) -> None:
        """V4-image model should instantiate without errors."""
        predictor = ImageTrafficPredictorV4()
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert len(predictor.feature_names) == 6  # 6 image-level features

    def test_v4_image_metadata_includes_proxy_label_note_or_falls_back(self) -> None:
        """V4-image metadata should expose the proxy label contract and fallback safely when absent."""
        metadata_path = ROOT / "model" / "saved_models" / "model_metadata_v4_image.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        note = metadata.get("proxy_label_note")
        if note is None:
            note = "V4-IMAGE traffic-density labels are proxy labels derived from vehicle-count thresholds and are not verified human ground-truth labels."

        assert "proxy" in note.lower()
        assert "not verified" in note.lower() or "not validated" in note.lower()
        assert "vehicle" in note.lower() or "threshold" in note.lower()


class TestImagePredictionOutput:
    """Test image prediction output format and correctness."""

    def test_image_prediction_returns_expected_keys(self) -> None:
        """Image prediction should return standard output structure."""
        img_path = ROOT / "data" / "images" / "raw" / "monikap-traffic-jam-2286005_1920.jpg.jpeg"
        if not img_path.exists():
            pytest.skip("Test image not found")

        predictor = ImageTrafficPredictor()
        result = predictor.predict_image(img_path)

        expected_keys = {
            "image",
            "overall_traffic_density",
            "overall_confidence",
            "total_vehicles",
            "counts",
            "probabilities",
            "model_version",
            "model_info",
            "proxy_label_note",
        }
        assert set(result.keys()) == expected_keys

    def test_image_prediction_label_is_valid(self) -> None:
        """Predicted label should be LOW, MEDIUM, or HIGH."""
        img_path = ROOT / "data" / "images" / "raw" / "monikap-traffic-jam-2286005_1920.jpg.jpeg"
        if not img_path.exists():
            pytest.skip("Test image not found")

        predictor = ImageTrafficPredictor()
        result = predictor.predict_image(img_path)

        assert result["overall_traffic_density"] in {"LOW", "MEDIUM", "HIGH"}
        assert 0.0 <= result["overall_confidence"] <= 1.0
        assert result["total_vehicles"] >= 0


class TestImageFormatSupport:
    """Ensure image format support hasn't changed."""

    def test_supported_extensions_include_jpg(self) -> None:
        """JPG/JPEG should be supported."""
        from model.inference.image_traffic_predictor import SUPPORTED_EXTENSIONS
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".jpeg" in SUPPORTED_EXTENSIONS

    def test_supported_extensions_include_png(self) -> None:
        """PNG should be supported."""
        from model.inference.image_traffic_predictor import SUPPORTED_EXTENSIONS
        assert ".png" in SUPPORTED_EXTENSIONS

    def test_supported_extensions_include_webp_and_bmp(self) -> None:
        """WEBP and BMP should be supported."""
        from model.inference.image_traffic_predictor import SUPPORTED_EXTENSIONS
        assert ".webp" in SUPPORTED_EXTENSIONS
        assert ".bmp" in SUPPORTED_EXTENSIONS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
