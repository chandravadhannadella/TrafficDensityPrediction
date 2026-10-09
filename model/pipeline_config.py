"""Central configuration for multi-video traffic dataset generation."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_VIDEO_DIR = PROJECT_ROOT / "data" / "videos" / "training"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "datasets" / "processed"
PREVIEW_VIDEO_DIR = PROJECT_ROOT / "outputs" / "processed_videos"

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
YOLO_MODEL = "yolov8n.pt"
FAST_DATASET_MODE = True
CONFIDENCE_THRESHOLD = 0.25
FRAME_SAMPLE_INTERVAL = 30
TIME_WINDOW_SECONDS = 10.0
YOLO_IMAGE_SIZE = 416
MAX_VIDEO_DURATION_SECONDS = 60.0
MAX_FRAMES_PER_VIDEO = 120
TRACKER_MAX_MISSED_FRAMES = 10
TRACKER_MAX_DISTANCE = 80.0
GENERATE_ANNOTATED_VIDEOS = False
