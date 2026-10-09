"""CLI entry point for reproducible multi-video traffic dataset generation."""

import logging
import argparse
from pathlib import Path
import pandas as pd

from dataset_builder import build_dataset, find_training_videos
from pipeline_config import PROCESSED_DATA_DIR


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def validate_dataset(frame_df: pd.DataFrame, window_df: pd.DataFrame) -> None:
    required_frame = {"video_id", "video_name", "frame_number", "total_vehicle_count"}
    required_window = {"video_id", "video_name", "window_start", "avg_total_vehicle_count"}
    if frame_df.empty or window_df.empty:
        raise ValueError("Generated dataset is empty")
    if not required_frame.issubset(frame_df.columns) or not required_window.issubset(window_df.columns):
        raise ValueError("Generated dataset is missing required columns")
    numeric = frame_df.select_dtypes(include="number")
    if numeric.isna().any().any() or (numeric < 0).any().any():
        raise ValueError("Generated frame dataset contains invalid numeric values")
    if frame_df.duplicated(subset=["video_id", "frame_number"]).any():
        raise ValueError("Generated frame dataset contains duplicates")
    if window_df.duplicated(subset=["video_id", "window_start"]).any():
        raise ValueError("Generated window dataset contains duplicates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, help="Process one video for validation")
    args = parser.parse_args()
    videos = find_training_videos()
    selected = [args.video] if args.video else videos
    if args.video and not args.video.is_absolute():
        selected = [Path(args.video)]
    def progress(video_index: int, video_total: int, frame: int, sampled: int, total_frames: int) -> None:
        percent = (frame / total_frames * 100) if total_frames else 0.0
        print(f"Current Video: {video_index} of {video_total} | Current Frame: {frame} | Sampled Frames Processed: {sampled} | Estimated Progress: {percent:.1f}%", flush=True)
    frame_df, window_df, manifest = build_dataset(selected, progress)
    validate_dataset(frame_df, window_df)
    successful = sum(entry.get("processing_status") == "success" for entry in manifest)
    print(f"Videos Found: {len(manifest)}")
    print(f"Videos Processed Successfully: {successful}")
    print(f"Videos Skipped: {len(manifest) - successful}")
    print(f"Sampled Frames: {len(frame_df)}")
    print(f"Total Time Windows Generated: {len(window_df)}")
    print(f"Total Dataset Rows: {len(window_df)}")
    print(f"Missing Values: {int(frame_df.isna().sum().sum() + window_df.isna().sum().sum())}")
    print(f"Duplicate Rows: {int(frame_df.duplicated().sum() + window_df.duplicated().sum())}")
    average = window_df["avg_total_vehicle_count"].mean() if not window_df.empty else 0.0
    print(f"Average Vehicles per Window: {average:.2f}")
    print("Dataset Saved Successfully:")
    print("data/datasets/processed/traffic_features.csv")


if __name__ == "__main__":
    main()
