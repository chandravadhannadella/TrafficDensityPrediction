from __future__ import annotations

import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from flask import current_app
from werkzeug.datastructures import FileStorage

from model.inference.video_traffic_predictor import VideoTrafficPredictor

JOBS: dict[str, dict[str, Any]] = {}


def _job_store_path() -> Path:
    root = Path(current_app.config["ROOT_DIR"]).resolve() if current_app else Path(__file__).resolve().parents[2]
    upload_dir = root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _annotated_video_path() -> Path:
    root = Path(current_app.config["ROOT_DIR"]).resolve() if current_app else Path(__file__).resolve().parents[2]
    annotated_dir = root / "data" / "videos" / "inference"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    return annotated_dir


def create_job_from_upload(file_storage: FileStorage, original_name: str) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    upload_dir = _job_store_path()
    safe_name = Path(original_name).name
    saved_path = upload_dir / safe_name
    file_storage.save(str(saved_path))

    job = {
        "job_id": job_id,
        "filename": safe_name,
        "file_path": str(saved_path),
        "status": "uploaded",
        "progress": 0,
        "result": None,
        "error": None,
    }
    JOBS[job_id] = job
    return job


def get_job(job_id: str) -> dict[str, Any] | None:
    return JOBS.get(job_id)


def update_job(job_id: str, **updates: Any) -> None:
    job = JOBS.get(job_id)
    if job is not None:
        job.update(updates)


def process_video_job(job_id: str) -> None:
    job = JOBS.get(job_id)
    if job is None:
        return

    def runner() -> None:
        try:
            update_job(job_id, status="processing", progress=5)
            predictor = VideoTrafficPredictor()
            
            # Generate annotated video
            annotated_dir = _annotated_video_path()
            original_filename = job["filename"]
            annotated_filename = Path(original_filename).stem + "_annotated.mp4"
            
            result = predictor.predict_video(
                job["file_path"], 
                generate_annotated_video=True, 
                annotated_output_dir=annotated_dir,
                progress_callback=lambda **kwargs: update_job(job_id, progress=min(95, int(kwargs.get("frames_processed", 0) * 100 / max(1, kwargs.get("frames_processed", 1)))))
            )
            
            # Copy original video to inference directory for browser-compatible preview
            # (annotated video uses mp4v codec which browsers don't support well)
            preview_filename = original_filename
            preview_path = annotated_dir / preview_filename
            shutil.copy2(job["file_path"], preview_path)
            
            # Add preview video filename to result
            result["preview_video_filename"] = preview_filename
            result["annotated_video_filename"] = annotated_filename
            
            update_job(job_id, status="completed", progress=100, result={
                "success": True,
                "job_id": job_id,
                "filename": job["filename"],
                "preview_video_filename": preview_filename,
                "annotated_video_filename": annotated_filename,
                "overall_traffic_density": result["overall_traffic_density"],
                "overall_confidence": result["overall_confidence"],
                "windows_processed": result["windows_processed"],
                "low_windows": result["low_windows"],
                "medium_windows": result["medium_windows"],
                "high_windows": result["high_windows"],
                "window_predictions": result["window_predictions"],
                "windows": result["window_predictions"],
                "proxy_label_note": "Baseline traffic-density estimator using proxy labels derived from avg_total_vehicle_count; not scientifically validated real-world ground truth.",
            })
        except Exception as exc:  # pragma: no cover - backend error path
            update_job(job_id, status="failed", progress=100, error=str(exc))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    update_job(job_id, status="processing", progress=10)
