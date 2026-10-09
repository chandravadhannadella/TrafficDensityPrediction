from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, send_from_directory

from backend.routes.prediction_routes import prediction_bp
from backend.routes.upload_routes import upload_bp

ROOT = Path(__file__).resolve().parent


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(ROOT / "templates"),
        static_folder=str(ROOT / "static"),
    )
    app.config["ROOT_DIR"] = str(ROOT)
    app.config["UPLOAD_FOLDER"] = str(ROOT / "uploads")
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False

    app.register_blueprint(upload_bp)
    app.register_blueprint(prediction_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    # Serve processed images
    @app.route("/data/images/processed/<path:filename>")
    def serve_processed_image(filename):
        return send_from_directory(str(ROOT / "data" / "images" / "processed"), filename)

    # Serve raw images
    @app.route("/data/images/raw/<path:filename>")
    def serve_raw_image(filename):
        return send_from_directory(str(ROOT / "data" / "images" / "raw"), filename)

    # Serve inference videos
    @app.route("/data/videos/inference/<path:filename>")
    def serve_inference_video(filename):
        return send_from_directory(str(ROOT / "data" / "videos" / "inference"), filename)

    # Serve uploaded videos
    @app.route("/uploads/videos/<path:filename>")
    def serve_uploaded_video(filename):
        return send_from_directory(str(ROOT / "uploads" / "videos"), filename)

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=True)
