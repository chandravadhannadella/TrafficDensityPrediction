from __future__ import annotations

import json
from pathlib import Path

from model.inference.video_traffic_predictor import VideoTrafficPredictor

ROOT = Path(__file__).resolve().parents[2]


def test_video_inference_pipeline() -> None:
    video_path = ROOT / "data" / "videos" / "testing" / "sample video (1).mp4"
    predictor = VideoTrafficPredictor()

    result = predictor.predict_video(video_path, generate_annotated_video=False)

    assert result["video"] == "sample video (1).mp4"
    assert result["windows_processed"] > 0
    assert result["overall_traffic_density"] in {"LOW", "MEDIUM", "HIGH"}
    assert 0.0 <= float(result["overall_confidence"]) <= 1.0
    assert len(result["window_predictions"]) == result["windows_processed"]

    json_path = ROOT / "data" / "results" / "video_prediction.json"
    csv_path = ROOT / "data" / "results" / "window_predictions.csv"

    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["video"] == "sample video (1).mp4"
    assert payload["overall_traffic_density"] == result["overall_traffic_density"]


if __name__ == "__main__":
    test_video_inference_pipeline()
    print("Inference pipeline check passed.")
