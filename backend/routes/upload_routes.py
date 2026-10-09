from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from backend.services.image_service import create_image_job_from_upload
from backend.services.video_service import create_job_from_upload

upload_bp = Blueprint("upload_bp", __name__)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@upload_bp.route("/api/upload", methods=["POST"])
def upload_video():
    file_storage = request.files.get("video")
    if file_storage is None or file_storage.filename == "":
        return jsonify({"success": False, "error": "No video file was uploaded."}), 400

    original_name = secure_filename(file_storage.filename)
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return jsonify({"success": False, "error": f"Unsupported video format: {suffix}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"}), 400

    job = create_job_from_upload(file_storage, original_name)
    return jsonify({
        "success": True,
        "status": "uploaded",
        "job_id": job["job_id"],
        "filename": job["filename"],
        "message": "Video uploaded successfully and is ready for processing.",
        "proxy_label_note": "The model uses proxy labels derived from aggregate traffic counts and is not a validated human ground-truth estimator.",
    })


@upload_bp.route("/api/upload-image", methods=["POST"])
def upload_image():
    file_storage = request.files.get("image")
    if file_storage is None or file_storage.filename == "":
        return jsonify({"success": False, "error": "No image file was uploaded."}), 400

    original_name = secure_filename(file_storage.filename)
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        return jsonify({"success": False, "error": f"Unsupported image format: {suffix}. Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"}), 400

    job = create_image_job_from_upload(file_storage, original_name)
    return jsonify({
        "success": True,
        "status": "uploaded",
        "job_id": job["job_id"],
        "filename": job["filename"],
        "message": "Image uploaded successfully and is ready for processing.",
        "proxy_label_note": "Single-image prediction maps detected counts to the saved traffic-density model contract and is experimental.",
    })
