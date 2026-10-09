from __future__ import annotations

from flask import Blueprint, jsonify

from backend.services.image_service import get_image_job, process_image_job
from backend.services.video_service import get_job, process_video_job

prediction_bp = Blueprint("prediction_bp", __name__)


@prediction_bp.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "service": "Traffic Density Prediction API",
        "proxy_label_note": "Baseline traffic-density estimates use proxy labels and should be treated as experimental, not verified ground truth.",
    })


@prediction_bp.route("/api/process/<job_id>", methods=["POST"])
def process_video(job_id: str):
    job = get_job(job_id)
    if job is None:
        return jsonify({"success": False, "error": "Job not found."}), 404

    if job["status"] == "completed":
        return jsonify({"success": True, "status": "completed", "job_id": job_id})

    if job["status"] == "failed":
        return jsonify({"success": False, "status": "failed", "job_id": job_id, "error": job.get("error")}), 400

    if job["status"] == "processing":
        return jsonify({"success": True, "status": "processing", "job_id": job_id})

    process_video_job(job_id)
    return jsonify({"success": True, "status": "processing", "job_id": job_id, "message": "Inference started."})


@prediction_bp.route("/api/process-image/<job_id>", methods=["POST"])
def process_image(job_id: str):
    job = get_image_job(job_id)
    if job is None:
        return jsonify({"success": False, "error": "Image job not found."}), 404

    if job["status"] == "completed":
        return jsonify({"success": True, "status": "completed", "job_id": job_id})

    if job["status"] == "failed":
        return jsonify({"success": False, "status": "failed", "job_id": job_id, "error": job.get("error")}), 400

    if job["status"] == "processing":
        return jsonify({"success": True, "status": "processing", "job_id": job_id})

    process_image_job(job_id)
    return jsonify({"success": True, "status": "processing", "job_id": job_id, "message": "Image inference started."})


@prediction_bp.route("/api/status/<job_id>", methods=["GET"])
def get_status(job_id: str):
    job = get_job(job_id)
    if job is not None:
        payload = {
            "job_id": job_id,
            "status": job["status"],
            "filename": job.get("filename"),
            "progress": job.get("progress", 0),
            "proxy_label_note": "This model uses proxy labels and is for baseline evaluation only.",
        }
        if job.get("error"):
            payload["error"] = job["error"]
        return jsonify(payload)

    image_job = get_image_job(job_id)
    if image_job is not None:
        payload = {
            "job_id": job_id,
            "status": image_job["status"],
            "filename": image_job.get("filename"),
            "progress": image_job.get("progress", 0),
            "proxy_label_note": "Single-image prediction maps counts to the saved traffic-density model contract and is experimental.",
        }
        if image_job.get("error"):
            payload["error"] = image_job["error"]
        return jsonify(payload)

    return jsonify({"success": False, "error": "Job not found."}), 404


@prediction_bp.route("/api/results/<job_id>", methods=["GET"])
def get_results(job_id: str):
    job = get_job(job_id)
    if job is not None:
        if job["status"] != "completed":
            return jsonify({"success": False, "status": job["status"], "error": "Inference has not completed yet."}), 409
        return jsonify({"success": True, **(job["result"] or {})})

    image_job = get_image_job(job_id)
    if image_job is not None:
        if image_job["status"] != "completed":
            return jsonify({"success": False, "status": image_job["status"], "error": "Inference has not completed yet."}), 409
        return jsonify({"success": True, **(image_job["result"] or {})})

    return jsonify({"success": False, "error": "Job not found."}), 404
