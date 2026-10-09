from __future__ import annotations

from pathlib import Path

from model.inference.video_traffic_predictor import VideoTrafficPredictor

ROOT = Path(__file__).resolve().parents[1]


def test_unseen_video_inference_pipeline() -> None:
    video_path = ROOT / "data" / "videos" / "inference" / "unseen_sample.mp4"
    predictor = VideoTrafficPredictor(model_path=ROOT / "model" / "saved_models" / "traffic_density_model.pkl")

    result = predictor.predict_video(video_path, generate_annotated_video=False)

    assert result["video"] == "unseen_sample.mp4"
    assert result["windows_processed"] > 0
    assert result["overall_traffic_density"] in {"LOW", "MEDIUM", "HIGH"}
    assert 0.0 <= float(result["overall_confidence"]) <= 1.0
    assert result["window_predictions"]
