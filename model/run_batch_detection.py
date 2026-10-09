"""Run vehicle detection on every supported image in a directory."""

from pathlib import Path

import cv2

from vehicle_detector import SUMMARY_LABELS, detect_vehicles


INPUT_DIR = Path("data/images/raw")
OUTPUT_DIR = Path("data/images/processed/batch")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        image_path
        for image_path in INPUT_DIR.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    print(f"Found {len(image_paths)} images")

    totals = {name: 0 for name in SUMMARY_LABELS}
    succeeded = 0
    failed = 0

    for image_path in image_paths:
        try:
            processed_image, counts, total = detect_vehicles(image_path)
            output_path = OUTPUT_DIR / f"{image_path.stem}_detected.jpg"
            if not cv2.imwrite(str(output_path), processed_image):
                raise OSError(f"Could not save processed image: {output_path}")
            for name, count in counts.items():
                totals[name] += count
            succeeded += 1
            print(f"{image_path.name}: {total} vehicles")
        except Exception as error:
            failed += 1
            print(f"{image_path.name}: ERROR: {error}")

    print(f"Completed: {succeeded} succeeded, {failed} failed")
    print("Aggregate counts:")
    for name, label in SUMMARY_LABELS.items():
        print(f"{label}: {totals[name]}")
    print(f"Total Vehicles: {sum(totals.values())}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
