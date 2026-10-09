"""Build reproducible combined traffic datasets and processing manifests."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from model.feature_extractor import FRAME_COLUMNS, aggregate_time_windows
from model.pipeline_config import (
    CONFIDENCE_THRESHOLD, FAST_DATASET_MODE, FRAME_SAMPLE_INTERVAL,
    GENERATE_ANNOTATED_VIDEOS,
    MAX_FRAMES_PER_VIDEO, MAX_VIDEO_DURATION_SECONDS, YOLO_IMAGE_SIZE,
    PREVIEW_VIDEO_DIR, PROCESSED_DATA_DIR, SUPPORTED_VIDEO_EXTENSIONS,
    TIME_WINDOW_SECONDS, TRACKER_MAX_DISTANCE, TRACKER_MAX_MISSED_FRAMES,
    TRAINING_VIDEO_DIR, YOLO_MODEL,
)
from model.video_processor import process_video

LOGGER = logging.getLogger(__name__)


def find_training_videos() -> list[Path]:
    return sorted(path for path in TRAINING_VIDEO_DIR.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS)


def build_dataset(video_paths: list[Path] | None = None, progress_callback: Any | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    videos = video_paths if video_paths is not None else find_training_videos()
    all_frames: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    processed = 0
    for index, video_path in enumerate(videos, start=1):
        LOGGER.info("Current Video: %d of %d: %s", index, len(videos), video_path.name)
        entry: dict[str, Any] = {"video_name": video_path.name, "video_path": str(video_path), "processing_date": datetime.now(timezone.utc).isoformat()}
        try:
            def report(frame_number: int, sampled_frames: int, total_frames: int, _duration: float) -> None:
                if progress_callback:
                    progress_callback(index, len(videos), frame_number, sampled_frames, total_frames)

            records, metadata = process_video(
                video_path, f"video_{index:04d}", YOLO_MODEL, CONFIDENCE_THRESHOLD,
                FRAME_SAMPLE_INTERVAL, TRACKER_MAX_DISTANCE, TRACKER_MAX_MISSED_FRAMES,
                YOLO_IMAGE_SIZE,
                MAX_VIDEO_DURATION_SECONDS if FAST_DATASET_MODE else None,
                MAX_FRAMES_PER_VIDEO if FAST_DATASET_MODE else None,
                GENERATE_ANNOTATED_VIDEOS if FAST_DATASET_MODE else GENERATE_ANNOTATED_VIDEOS,
                PREVIEW_VIDEO_DIR, report,
            )
            all_frames.extend(records)
            entry.update(metadata, generated_windows=0, processing_status="success")
            processed += 1
        except Exception as error:
            LOGGER.exception("Skipping video %s", video_path.name)
            entry.update(processing_status="skipped", error=str(error), sampled_frames=0, generated_windows=0)
        manifest.append(entry)
    frame_df = pd.DataFrame(all_frames, columns=FRAME_COLUMNS)
    if not frame_df.empty:
        frame_df = frame_df.dropna(subset=["video_id", "frame_number"]).drop_duplicates(subset=["video_id", "frame_number"])
        numeric_columns = frame_df.select_dtypes(include="number").columns
        frame_df = frame_df[frame_df[numeric_columns].notna().all(axis=1)]
        count_columns = [f"{name}_count" for name in ("car", "bus", "truck", "motorcycle", "bicycle")] + ["total_vehicle_count", "unique_vehicle_count"]
        frame_df = frame_df[(frame_df[count_columns] >= 0).all(axis=1)]
    window_df = aggregate_time_windows(frame_df.to_dict("records"), TIME_WINDOW_SECONDS)
    if not window_df.empty:
        window_df = window_df.drop_duplicates(subset=["video_id", "window_start"]).reset_index(drop=True)
    for entry in manifest:
        entry["generated_windows"] = int((window_df["video_name"] == entry["video_name"]).sum()) if not window_df.empty else 0
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame_df.to_csv(PROCESSED_DATA_DIR / "frame_features.csv", index=False)
    window_df.to_csv(PROCESSED_DATA_DIR / "traffic_features.csv", index=False)
    (PROCESSED_DATA_DIR / "processing_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Videos found: %d; processed: %d; skipped: %d", len(videos), processed, len(videos) - processed)
    return frame_df, window_df, manifest
