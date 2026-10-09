from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.inference.video_traffic_predictor import VideoTrafficPredictor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run traffic-density inference on a new video using the saved baseline model.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--sample-interval", type=int, default=30, help="Frame sampling interval.")
    parser.add_argument("--max-duration", type=float, default=None, help="Maximum video duration in seconds.")
    parser.add_argument("--window-size", type=float, default=10.0, help="Inference time-window size in seconds.")
    parser.add_argument("--annotated", action="store_true", help="Generate an annotated output video.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        predictor = VideoTrafficPredictor(
            window_seconds=args.window_size,
            sample_interval=args.sample_interval,
        )
        result = predictor.predict_video(
            args.video,
            max_duration_seconds=args.max_duration,
            generate_annotated_video=args.annotated,
        )
        print(json.dumps({
            "video": result["video"],
            "overall_traffic_density": result["overall_traffic_density"],
            "overall_confidence": result["overall_confidence"],
            "windows_processed": result["windows_processed"],
            "low_windows": result["low_windows"],
            "medium_windows": result["medium_windows"],
            "high_windows": result["high_windows"],
        }, indent=2))
    except Exception as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
