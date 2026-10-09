"""Feature extraction and time-window aggregation for traffic observations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from model.vehicle_tracker import Track


CLASS_NAMES = ("car", "bus", "truck", "motorcycle", "bicycle")
FRAME_COLUMNS = [
    "video_id", "video_name", "frame_number", "timestamp_seconds",
    "car_count", "bus_count", "truck_count", "motorcycle_count", "bicycle_count",
    "total_vehicle_count", "unique_vehicle_count", "average_vehicle_confidence",
    "average_vehicle_movement", "approximate_average_speed", "traffic_flow",
    "congestion_score",
]


def extract_frame_features(
    video_id: str,
    video_name: str,
    frame_number: int,
    timestamp_seconds: float,
    tracks: list[Track],
    fps: float,
    sample_interval: int,
) -> dict[str, Any]:
    counts = {class_name: 0 for class_name in CLASS_NAMES}
    for track in tracks:
        counts[track.class_name] += 1
    total = len(tracks)
    unique_ids = {track.track_id for track in tracks}
    average_confidence = sum(track.confidence for track in tracks) / total if total else 0.0
    average_movement = sum(track.distance for track in tracks) / total if total else 0.0
    sampled_seconds = sample_interval / fps if fps > 0 else 0.0
    speed = average_movement / sampled_seconds if sampled_seconds > 0 else 0.0
    return {
        "video_id": video_id,
        "video_name": video_name,
        "frame_number": frame_number,
        "timestamp_seconds": round(timestamp_seconds, 3),
        **{f"{name}_count": counts[name] for name in CLASS_NAMES},
        "total_vehicle_count": total,
        "unique_vehicle_count": len(unique_ids),
        "average_vehicle_confidence": round(average_confidence, 4),
        "average_vehicle_movement": round(average_movement, 4),
        "approximate_average_speed": round(speed, 4),
        "traffic_flow": round(len(unique_ids) / sampled_seconds, 4) if sampled_seconds else 0.0,
        "congestion_score": round(min(total / 20.0, 1.0), 4),
    }


def aggregate_time_windows(frame_records: list[dict[str, Any]], window_seconds: float) -> pd.DataFrame:
    if not frame_records:
        return pd.DataFrame()
    frame_data = pd.DataFrame(frame_records)
    frame_data["window_index"] = (frame_data["timestamp_seconds"] // window_seconds).astype(int)
    rows: list[dict[str, Any]] = []
    grouped = frame_data.groupby(["video_id", "video_name", "window_index"], sort=True)
    for (video_id, video_name, window_index), group in grouped:
        row: dict[str, Any] = {
            "video_id": video_id,
            "video_name": video_name,
            "window_start": round(window_index * window_seconds, 3),
            "window_end": round((window_index + 1) * window_seconds, 3),
        }
        for class_name in CLASS_NAMES:
            row[f"avg_{class_name}_count"] = round(group[f"{class_name}_count"].mean(), 4)
        row["avg_total_vehicle_count"] = round(group["total_vehicle_count"].mean(), 4)
        row["max_vehicle_count"] = int(group["total_vehicle_count"].max())
        row["min_vehicle_count"] = int(group["total_vehicle_count"].min())
        row["unique_vehicle_count"] = int(group["unique_vehicle_count"].max())
        row["average_vehicle_confidence"] = round(group["average_vehicle_confidence"].mean(), 4)
        row["average_vehicle_movement"] = round(group["average_vehicle_movement"].mean(), 4)
        row["approximate_average_speed"] = round(group["approximate_average_speed"].mean(), 4)
        row["traffic_flow"] = round(group["traffic_flow"].mean(), 4)
        row["congestion_score"] = round(group["congestion_score"].mean(), 4)
        rows.append(row)
    return pd.DataFrame(rows)
