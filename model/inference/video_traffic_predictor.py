from __future__ import annotations

import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from model.feature_extractor import aggregate_time_windows, extract_frame_features
from model.inference.model_config import NoOpScaler, resolve_model_paths
from model.pipeline_config import CONFIDENCE_THRESHOLD, FRAME_SAMPLE_INTERVAL, TIME_WINDOW_SECONDS, TRACKER_MAX_DISTANCE, TRACKER_MAX_MISSED_FRAMES, YOLO_IMAGE_SIZE, YOLO_MODEL
from model.vehicle_detector import detect_frame
from model.vehicle_tracker import CentroidTracker

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MODEL_DIR = Path(__file__).resolve().parents[1] / "saved_models"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "results"


class VideoTrafficPredictor:
    """Process a new video into traffic-density predictions using the saved baseline model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        feature_names_path: str | Path | None = None,
        window_seconds: float = TIME_WINDOW_SECONDS,
        sample_interval: int = FRAME_SAMPLE_INTERVAL,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        image_size: int = YOLO_IMAGE_SIZE,
    ) -> None:
        if model_path is None and scaler_path is None and feature_names_path is None:
            model_path, scaler_path, feature_names_path = resolve_model_paths("v3")

        self.model_path = Path(model_path) if model_path is not None else MODEL_DIR / "traffic_density_model.pkl"
        self.scaler_path = Path(scaler_path) if scaler_path is not None else MODEL_DIR / "scaler.pkl"
        self.feature_names_path = Path(feature_names_path) if feature_names_path is not None else MODEL_DIR / "feature_names.json"
        self.window_seconds = float(window_seconds)
        self.sample_interval = int(sample_interval)
        self.confidence_threshold = float(confidence_threshold)
        self.image_size = int(image_size)

        print(f"MODEL VERSION: V3")
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
        return VideoTrafficPredictor._load_pickle(path)

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
    def _valid_video_path(video_path: str | Path) -> Path:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported video format: {path.suffix}. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}")
        return path

    def _prepare_window_features(self, window_df: pd.DataFrame) -> pd.DataFrame:
        if window_df.empty:
            raise ValueError("No valid windows were produced from the video.")

        missing = [name for name in self.feature_names if name not in window_df.columns]
        if missing:
            raise ValueError(
                "Missing inference features required by training metadata: "
                + ", ".join(missing)
                + ". Feature order must match the saved training feature list."
            )

        prepared = window_df[[*self.feature_names]].copy()
        if len(prepared.columns) != len(self.feature_names):
            raise ValueError("Inference feature count does not match the saved training feature list.")
        return prepared

    def _predict_window(self, window_df: pd.DataFrame) -> dict[str, Any]:
        X = self._prepare_window_features(window_df)
        scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(scaled)[0]
        predicted_label = self.model.classes_[np.argmax(probabilities)]
        confidence = float(np.max(probabilities))
        class_to_prob = {label: float(probabilities[index]) for index, label in enumerate(self.model.classes_)}
        return {
            "traffic_density": str(predicted_label),
            "confidence": confidence,
            "probabilities": {key: float(class_to_prob.get(key, 0.0)) for key in self.label_order if key in class_to_prob},
        }

    def _aggregate_video_result(self, window_rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not window_rows:
            raise ValueError("No valid prediction windows were produced for the video.")

        class_counts = Counter(row["predicted_traffic_density"] for row in window_rows)
        dominant_label, _ = class_counts.most_common(1)[0]

        all_probs = []
        for row in window_rows:
            probabilities = row.get("probabilities", {})
            if probabilities:
                all_probs.append(probabilities)

        if not all_probs:
            overall_confidence = 0.0
        else:
            mean_prob = {label: float(np.mean([item.get(label, 0.0) for item in all_probs])) for label in self.label_order}
            overall_confidence = float(mean_prob.get(dominant_label, 0.0))

        low_count = sum(1 for row in window_rows if row["predicted_traffic_density"] == "LOW")
        medium_count = sum(1 for row in window_rows if row["predicted_traffic_density"] == "MEDIUM")
        high_count = sum(1 for row in window_rows if row["predicted_traffic_density"] == "HIGH")

        return {
            "overall_traffic_density": dominant_label,
            "overall_confidence": round(overall_confidence, 4),
            "windows_processed": len(window_rows),
            "low_windows": low_count,
            "medium_windows": medium_count,
            "high_windows": high_count,
        }

    def predict_video(
        self,
        video_path: str | Path,
        *,
        max_duration_seconds: float | None = None,
        max_sampled_frames: int | None = None,
        generate_annotated_video: bool = False,
        annotated_output_dir: str | Path | None = None,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        path = self._valid_video_path(video_path)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Could not open or decode video: {path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        tracker = CentroidTracker(TRACKER_MAX_DISTANCE, TRACKER_MAX_MISSED_FRAMES)
        frame_records: list[dict[str, Any]] = []
        sampled_frames = 0
        frame_index = 0
        window_groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
        annotated_writer = None
        output_annotation_path = None

        if generate_annotated_video:
            output_dir = Path(annotated_output_dir) if annotated_output_dir else RESULTS_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            output_annotation_path = output_dir / f"{path.stem}_annotated.mp4"
            writer = cv2.VideoWriter(
                str(output_annotation_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps or 25.0,
                (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            )
            annotated_writer = writer

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if max_duration_seconds is not None and frame_index / max(fps, 1.0) > max_duration_seconds:
                    break
                if frame_index % self.sample_interval == 0:
                    detections = detect_frame(frame, YOLO_MODEL, self.confidence_threshold, self.image_size)
                    if not detections:
                        frame_index += 1
                        continue
                    tracks = tracker.update(detections)
                    record = extract_frame_features(
                        video_id=path.stem,
                        video_name=path.name,
                        frame_number=frame_index,
                        timestamp_seconds=float(frame_index / max(fps, 1.0)),
                        tracks=tracks,
                        fps=fps,
                        sample_interval=self.sample_interval,
                    )
                    frame_records.append(record)
                    sampled_frames += 1
                    window_start = (record["timestamp_seconds"] // self.window_seconds) * self.window_seconds
                    window_end = window_start + self.window_seconds
                    window_groups.setdefault((window_start, window_end), []).append(record)
                    if progress_callback:
                        progress_callback(
                            video_name=path.name,
                            frames_processed=sampled_frames,
                            current_time=record["timestamp_seconds"],
                            current_density="UNKNOWN",
                            confidence=0.0,
                        )
                    if generate_annotated_video and annotated_writer is not None:
                        for track in tracks:
                            x1, y1, x2, y2 = track.bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
                            cv2.putText(frame, f"{track.class_name} #{track.track_id}", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Vehicles: {len(tracks)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        annotated_writer.write(frame)
                frame_index += 1
                if max_sampled_frames is not None and sampled_frames >= max_sampled_frames:
                    break
        finally:
            capture.release()
            if annotated_writer is not None:
                annotated_writer.release()

        if not frame_records:
            raise ValueError("No detectable vehicles were found in the video. The input video may not contain enough traffic or may not be suitable for inference.")

        aggregated = aggregate_time_windows(frame_records, self.window_seconds)
        if aggregated.empty:
            raise ValueError("No valid time windows were generated from the video. Try increasing the sample interval or processing a longer clip.")

        window_predictions: list[dict[str, Any]] = []
        for _, row in aggregated.iterrows():
            row_dict = row.to_dict()
            X = self._prepare_window_features(pd.DataFrame([row_dict]))
            if X.empty:
                continue
            scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(scaled)[0]
            label_index = int(np.argmax(probabilities))
            label = str(self.model.classes_[label_index])
            confidence = float(probabilities[label_index])
            row_result = {
                "video_name": path.name,
                "window_start": float(row_dict.get("window_start", 0.0)),
                "window_end": float(row_dict.get("window_end", self.window_seconds)),
                "avg_car_count": float(row_dict.get("avg_car_count", 0.0)),
                "avg_bus_count": float(row_dict.get("avg_bus_count", 0.0)),
                "avg_truck_count": float(row_dict.get("avg_truck_count", 0.0)),
                "avg_motorcycle_count": float(row_dict.get("avg_motorcycle_count", 0.0)),
                "avg_bicycle_count": float(row_dict.get("avg_bicycle_count", 0.0)),
                "avg_vehicle_confidence": float(row_dict.get("average_vehicle_confidence", 0.0)),
                "average_vehicle_movement": float(row_dict.get("average_vehicle_movement", 0.0)),
                "approximate_average_speed": float(row_dict.get("approximate_average_speed", 0.0)),
                "traffic_flow": float(row_dict.get("traffic_flow", 0.0)),
                "predicted_traffic_density": label,
                "confidence": round(confidence, 4),
                "probabilities": {class_name: round(float(probabilities[i]), 4) for i, class_name in enumerate(self.model.classes_)},
            }
            window_predictions.append(row_result)

        if not window_predictions:
            raise ValueError("Model prediction failed on all generated windows.")

        summary = self._aggregate_video_result(window_predictions)
        summary["video"] = path.name
        summary["window_predictions"] = window_predictions

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_output = {
            "video": path.name,
            "overall_traffic_density": summary["overall_traffic_density"],
            "overall_confidence": summary["overall_confidence"],
            "windows_processed": summary["windows_processed"],
            "low_windows": summary["low_windows"],
            "medium_windows": summary["medium_windows"],
            "high_windows": summary["high_windows"],
            "window_predictions": window_predictions,
            "proxy_label_note": "Baseline traffic-density estimator using proxy labels derived from avg_total_vehicle_count; not scientifically validated real-world ground truth.",
        }
        (RESULTS_DIR / "video_prediction.json").write_text(json.dumps(json_output, indent=2), encoding="utf-8")

        csv_columns = [
            "video_name",
            "window_start",
            "window_end",
            "avg_car_count",
            "avg_bus_count",
            "avg_truck_count",
            "avg_motorcycle_count",
            "avg_bicycle_count",
            "avg_vehicle_confidence",
            "average_vehicle_movement",
            "approximate_average_speed",
            "traffic_flow",
            "predicted_traffic_density",
            "confidence",
        ]
        with (RESULTS_DIR / "window_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_columns)
            writer.writeheader()
            for row in window_predictions:
                writer.writerow({
                    "video_name": row["video_name"],
                    "window_start": row["window_start"],
                    "window_end": row["window_end"],
                    "avg_car_count": row["avg_car_count"],
                    "avg_bus_count": row["avg_bus_count"],
                    "avg_truck_count": row["avg_truck_count"],
                    "avg_motorcycle_count": row["avg_motorcycle_count"],
                    "avg_bicycle_count": row["avg_bicycle_count"],
                    "avg_vehicle_confidence": row["avg_vehicle_confidence"],
                    "average_vehicle_movement": row["average_vehicle_movement"],
                    "approximate_average_speed": row["approximate_average_speed"],
                    "traffic_flow": row["traffic_flow"],
                    "predicted_traffic_density": row["predicted_traffic_density"],
                    "confidence": row["confidence"],
                })

        return {
            "video": path.name,
            "overall_traffic_density": summary["overall_traffic_density"],
            "overall_confidence": summary["overall_confidence"],
            "windows_processed": summary["windows_processed"],
            "low_windows": summary["low_windows"],
            "medium_windows": summary["medium_windows"],
            "high_windows": summary["high_windows"],
            "window_predictions": window_predictions,
        }
