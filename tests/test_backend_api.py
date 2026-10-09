from __future__ import annotations

from pathlib import Path

from app import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_app_health_and_upload() -> None:
    app = create_app()
    client = app.test_client()

    health = client.get("/api/health")
    assert health.status_code == 200
    payload = health.get_json()
    assert payload["status"] == "ok"
    assert payload["service"] == "Traffic Density Prediction API"

    bad = client.post("/api/upload", data={"video": (b"bad", "bad.txt")}, content_type="multipart/form-data")
    assert bad.status_code == 400
    assert bad.get_json()["success"] is False

    sample = ROOT / "data" / "videos" / "inference" / "unseen_sample.mp4"
    upload = client.post(
        "/api/upload",
        data={"video": (sample.open("rb"), sample.name)},
        content_type="multipart/form-data",
    )
    assert upload.status_code == 200
    upload_json = upload.get_json()
    assert upload_json["success"] is True
    assert upload_json["status"] == "uploaded"
    assert upload_json["job_id"]


def test_process_and_results_flow() -> None:
    app = create_app()
    client = app.test_client()
    sample = ROOT / "data" / "videos" / "inference" / "unseen_sample.mp4"

    upload = client.post(
        "/api/upload",
        data={"video": (sample.open("rb"), sample.name)},
        content_type="multipart/form-data",
    )
    job_id = upload.get_json()["job_id"]

    process_response = client.post(f"/api/process/{job_id}")
    assert process_response.status_code == 200
    assert process_response.get_json()["success"] is True

    for _ in range(80):
        status = client.get(f"/api/status/{job_id}")
        payload = status.get_json()
        if payload.get("status") in {"completed", "failed"}:
            break
        import time
        time.sleep(0.5)

    assert status.status_code == 200
    status_payload = status.get_json()
    assert status_payload["job_id"] == job_id
    assert status_payload["status"] in {"completed", "failed"}

    if status_payload["status"] == "failed":
        assert "error" in status_payload
    else:
        results = client.get(f"/api/results/{job_id}")
        assert results.status_code == 200
        result_json = results.get_json()
        assert result_json["success"] is True
        assert result_json["overall_traffic_density"] in {"LOW", "MEDIUM", "HIGH"}
        assert "windows" in result_json


def test_image_upload_and_processing_flow() -> None:
    app = create_app()
    client = app.test_client()
    sample = ROOT / "data" / "images" / "raw" / "images_231.jpg"

    upload = client.post(
        "/api/upload-image",
        data={"image": (sample.open("rb"), sample.name)},
        content_type="multipart/form-data",
    )
    assert upload.status_code == 200
    payload = upload.get_json()
    assert payload["success"] is True
    assert payload["status"] == "uploaded"
    assert payload["job_id"]

    job_id = payload["job_id"]
    process_response = client.post(f"/api/process-image/{job_id}")
    assert process_response.status_code == 200
    assert process_response.get_json()["success"] is True

    for _ in range(40):
        status = client.get(f"/api/status/{job_id}")
        if status.get_json().get("status") in {"completed", "failed"}:
            break
        import time
        time.sleep(0.25)

    final_status = client.get(f"/api/status/{job_id}")
    status_payload = final_status.get_json()
    assert status_payload["status"] in {"completed", "failed"}
    if status_payload["status"] == "failed":
        assert "error" in status_payload
    else:
        results = client.get(f"/api/results/{job_id}")
        assert results.status_code == 200
        result_json = results.get_json()
        assert result_json["success"] is True
        assert result_json["overall_traffic_density"] in {"LOW", "MEDIUM", "HIGH"}
        assert "image" in result_json
