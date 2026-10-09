from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Any

import cv2
from flask import current_app
from werkzeug.datastructures import FileStorage

from model.inference.image_traffic_predictor import ImageTrafficPredictor
from model.vehicle_detector import detect_vehicles

IMAGE_JOBS: dict[str, dict[str, Any]] = {}


def _image_job_store_path() -> Path:
    root = Path(current_app.config["ROOT_DIR"]).resolve() if current_app else Path(__file__).resolve().parents[2]
    image_dir = root / "uploads" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


def _processed_image_path() -> Path:
    root = Path(current_app.config["ROOT_DIR"]).resolve() if current_app else Path(__file__).resolve().parents[2]
    processed_dir = root / "data" / "images" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def create_image_job_from_upload(file_storage: FileStorage, original_name: str) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    upload_dir = _image_job_store_path()
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
    IMAGE_JOBS[job_id] = job
    return job


def get_image_job(job_id: str) -> dict[str, Any] | None:
    return IMAGE_JOBS.get(job_id)


def update_image_job(job_id: str, **updates: Any) -> None:
    job = IMAGE_JOBS.get(job_id)
    if job is not None:
        job.update(updates)


def process_image_job(job_id: str) -> None:
    job = IMAGE_JOBS.get(job_id)
    if job is None:
        return

    def runner() -> None:
        try:
            update_image_job(job_id, status="processing", progress=10)
            predictor = ImageTrafficPredictor()
            
            # Detect vehicles and get processed image with bounding boxes
            processed_image, counts, total = detect_vehicles(
                job["file_path"], 
                model_name="yolov8n.pt", 
                confidence=0.25, 
                image_size=640
            )
            
            # Save processed image to data/images/processed/
            processed_dir = _processed_image_path()
            processed_filename = job["filename"]
            processed_path = processed_dir / processed_filename
            cv2.imwrite(str(processed_path), processed_image)
            
            # Now run the prediction
            result = predictor.predict_image(job["file_path"])
            update_image_job(job_id, status="completed", progress=100, result={
                "success": True,
                "job_id": job_id,
                "filename": job["filename"],
                "image": result["image"],
                "overall_traffic_density": result["overall_traffic_density"],
                "overall_confidence": result["overall_confidence"],
                "total_vehicles": result["total_vehicles"],
                "counts": result["counts"],
                "probabilities": result["probabilities"],
                "model_version": result.get("model_version") or "v4_image",
                "model_info": result.get("model_info") or "V4-IMAGE: dedicated image-specific model trained on 924 raw images with proxy labels",
                "proxy_label_note": result.get("proxy_label_note") or (
                    "V4-IMAGE traffic-density labels are proxy labels derived from total_vehicle_count thresholds "
                    "(LOW <= 8, MEDIUM 9-15, HIGH >= 16) and are not verified human ground-truth labels."
                ),
            })
        except Exception as exc:  # pragma: no cover - backend error path
            update_image_job(job_id, status="failed", progress=100, error=str(exc))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    update_image_job(job_id, status="processing", progress=15)
