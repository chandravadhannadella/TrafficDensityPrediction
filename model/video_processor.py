"""Process one traffic video into sampled, tracked frame features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2

from model.feature_extractor import extract_frame_features
from model.vehicle_detector import detect_frame
from model.vehicle_tracker import CentroidTracker

LOGGER = logging.getLogger(__name__)


def process_video(
    video_path: str | Path,
    video_id: str,
    model_name: str,
    confidence: float,
    sample_interval: int,
    max_distance: float,
    max_missed_frames: int,
    image_size: int,
    max_duration_seconds: float | None = None,
    max_sampled_frames: int | None = None,
    generate_preview: bool = False,
    preview_dir: str | Path | None = None,
    progress_callback: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open or decode video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0
    duration_limit = max_duration_seconds if max_duration_seconds and max_duration_seconds > 0 else None
    frame_limit = max_sampled_frames if max_sampled_frames and max_sampled_frames > 0 else None
    capped = bool((duration_limit and duration > duration_limit) or frame_limit)
    tracker = CentroidTracker(max_distance, max_missed_frames)
    records: list[dict[str, Any]] = []
    writer = None
    sampled_frames = 0
    frame_number = 0
    try:
        if generate_preview and preview_dir:
            output_path = Path(preview_dir) / f"{video_path.stem}_annotated.mp4"
            Path(preview_dir).mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps or 25.0, (width, height)
            )
        while True:
            success, frame = capture.read()
            if not success:
                break
            timestamp = frame_number / fps if fps else 0.0
            if duration_limit and timestamp >= duration_limit:
                break
            if frame_number % sample_interval == 0 and (frame_limit is None or sampled_frames < frame_limit):
                detections = detect_frame(frame, model_name, confidence, image_size)
                tracks = tracker.update(detections)
                records.append(
                    extract_frame_features(
                        video_id, video_path.name, frame_number, timestamp,
                        tracks, fps, sample_interval,
                    )
                )
                sampled_frames += 1
                if progress_callback:
                    progress_callback(frame_number, sampled_frames, frame_count, duration)
                if frame_limit is not None and sampled_frames >= frame_limit:
                    break
                if writer is not None:
                    for track in tracks:
                        x1, y1, x2, y2 = track.bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
                        cv2.putText(frame, f"{track.class_name} #{track.track_id}", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Vehicles: {len(tracks)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    writer.write(frame)
            frame_number += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    metadata = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "sampled_frames": sampled_frames,
        "capped": capped and (frame_number < frame_count or (frame_limit is not None and sampled_frames >= frame_limit)),
        "processed_frame_count": frame_number,
    }
    if metadata["capped"]:
        LOGGER.warning(
            "Video capped: %s (max duration=%s seconds, max sampled frames=%s)",
            video_path.name,
            max_duration_seconds,
            max_sampled_frames,
        )
    return records, metadata
