"""Single-image vehicle detection with a pre-trained Ultralytics YOLO model."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import TypedDict
import cv2
import numpy as np
from ultralytics import YOLO


VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
SUMMARY_LABELS = {
    "car": "Cars",
    "bus": "Buses",
    "truck": "Trucks",
    "motorcycle": "Motorcycles",
    "bicycle": "Bicycles",
}


class VehicleDetection(TypedDict):
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> YOLO:
    try:
        return YOLO(model_name)
    except Exception as error:
        raise RuntimeError(
            f"Unable to load YOLO model '{model_name}'. "
            "Check the model name and your network or local weights."
        ) from error


def _draw_summary(image: np.ndarray, counts: Counter[str], total: int) -> None:
    lines = [
        f"{SUMMARY_LABELS[name]}: {counts[name]}"
        for name in ("car", "bus", "truck", "motorcycle", "bicycle")
    ]
    lines.append(f"Total Vehicles: {total}")

    panel_height = 20 + len(lines) * 26
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (260, panel_height), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (20, 36 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def detect_frame(
    frame: np.ndarray,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    image_size: int = 640,
) -> list[VehicleDetection]:
    """Return traffic-vehicle detections for a BGR frame."""
    if frame is None or frame.size == 0:
        raise ValueError("Unsupported or empty video frame")

    model = _load_model(model_name)
    try:
        results = model.predict(source=frame, conf=confidence, imgsz=image_size, verbose=False)
    except Exception as error:
        raise RuntimeError("YOLO detection failed for a video frame") from error

    detections: list[VehicleDetection] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            vehicle_name = VEHICLE_CLASSES.get(int(box.cls[0]))
            if vehicle_name is None:
                continue
            detections.append(
                {
                    "class_name": vehicle_name,
                    "confidence": float(box.conf[0]),
                    "bbox": tuple(map(int, box.xyxy[0].tolist())),
                }
            )
    return detections


def detect_vehicles(
    image_path: str | Path,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    image_size: int = 640,
) -> tuple[np.ndarray, dict[str, int], int]:
    """Detect vehicles in one image and return its annotation and counts.

    The returned image is a BGR NumPy array, ready for ``cv2.imwrite`` or
    later reuse as a video-frame detector.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Traffic image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unsupported or corrupted image: {image_path}")

    detections = detect_frame(image, model_name=model_name, confidence=confidence, image_size=image_size)

    counts: Counter[str] = Counter()
    processed_image = image.copy()

    for detection in detections:
        vehicle_name = detection["class_name"]
        score = detection["confidence"]
        x1, y1, x2, y2 = detection["bbox"]
        counts[vehicle_name] += 1
        cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 220, 80), 2)
        label = f"{vehicle_name.title()} {score:.2f}"
        text_y = max(y1 - 8, 20)
        cv2.putText(
            processed_image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 80),
            2,
            cv2.LINE_AA,
        )

    total = sum(counts.values())
    complete_counts = {name: counts[name] for name in SUMMARY_LABELS}
    _draw_summary(processed_image, Counter(complete_counts), total)
    return processed_image, complete_counts, total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect vehicles in a traffic image")
    parser.add_argument(
        "image",
        nargs="?",
        default="data/images/raw/traffic.jpg",
        help="Path to the input traffic image",
    )
    parser.add_argument(
        "--output",
        default="data/images/processed/detected_traffic.jpg",
        help="Path for the annotated output image",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model or local weights path")
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        processed_image, counts, total = detect_vehicles(
            args.image,
            model_name=args.model,
            confidence=args.confidence,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), processed_image):
            raise OSError(f"Could not save processed image: {output_path}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print("Vehicle detection summary")
    for name, count in counts.items():
        print(f"{SUMMARY_LABELS[name]}: {count}")
    print(f"Total Vehicles: {total}")
    print(f"Processed image saved to: {output_path}")


if __name__ == "__main__":
    main()
