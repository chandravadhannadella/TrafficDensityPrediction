"""Prepare the final traffic-density dataset and report.

This script intentionally does not train a model. It validates the processed time-window
features, assigns a traffic_density label via a documented proxy-labeling workflow,
performs a video-group split, and writes the train/validation/test CSVs and summary report.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from traffic_labeler import TrafficLabeler, detect_compatible_csv_datasets, save_dataset_integration_report

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "datasets" / "processed"
FINAL_DIR = ROOT / "data" / "datasets" / "final"
RAW_DIR = ROOT / "data" / "datasets" / "raw"

FEATURE_COLUMNS = [
    "avg_car_count",
    "avg_bus_count",
    "avg_truck_count",
    "avg_motorcycle_count",
    "avg_bicycle_count",
    "avg_total_vehicle_count",
    "max_vehicle_count",
    "unique_vehicle_count",
    "traffic_flow",
]
TARGET_COLUMN = "traffic_density"


def _build_labeled_dataset() -> pd.DataFrame:
    source = PROCESSED_DIR / "traffic_features.csv"
    traffic_df = pd.read_csv(source)
    labeler = TrafficLabeler(low_threshold=4.0, medium_threshold=10.0)
    labeled_df = labeler.label_dataframe(traffic_df)
    labeled_df = labeled_df.drop_duplicates().reset_index(drop=True)
    labeled_df.to_csv(PROCESSED_DIR / "labeled_traffic_features.csv", index=False)
    return labeled_df


def _group_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    unique_videos = sorted(df["video_id"].unique().tolist())
    rng = random.Random(42)
    shuffled = unique_videos[:]
    rng.shuffle(shuffled)

    train_count = 7
    val_count = 2
    test_count = len(shuffled) - train_count - val_count
    if test_count < 1:
        raise ValueError("Not enough videos for a valid train/validation/test split.")

    train_videos = shuffled[:train_count]
    val_videos = shuffled[train_count:train_count + val_count]
    test_videos = shuffled[train_count + val_count:]

    train_df = df[df["video_id"].isin(train_videos)].copy().reset_index(drop=True)
    val_df = df[df["video_id"].isin(val_videos)].copy().reset_index(drop=True)
    test_df = df[df["video_id"].isin(test_videos)].copy().reset_index(drop=True)

    split_map = {
        "train": sorted(train_videos),
        "validation": sorted(val_videos),
        "test": sorted(test_videos),
    }
    return train_df, val_df, test_df, split_map


def _export_split_frames(df: pd.DataFrame) -> pd.DataFrame:
    return df[[*FEATURE_COLUMNS, TARGET_COLUMN]].copy().reset_index(drop=True)


def _build_report(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, split_map: dict[str, list[str]]) -> dict:
    all_labeled = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_export = _export_split_frames(all_labeled)
    target_distribution = all_export[TARGET_COLUMN].value_counts().sort_index().to_dict()
    report = {
        "dataset_summary": {
            "total_rows": int(len(all_labeled)),
            "total_videos": int(all_labeled["video_id"].nunique()),
            "total_features": int(len(FEATURE_COLUMNS)),
            "target_column": TARGET_COLUMN,
            "target_classes": ["LOW", "MEDIUM", "HIGH"],
            "class_distribution": {k: int(v) for k, v in target_distribution.items()},
            "missing_values": {column: int(all_export[column].isna().sum()) for column in all_export.columns},
            "duplicate_rows": int(all_export.duplicated().sum()),
            "feature_ranges": {
                column: {
                    "min": float(all_export[column].min()) if pd.api.types.is_numeric_dtype(all_export[column]) else None,
                    "max": float(all_export[column].max()) if pd.api.types.is_numeric_dtype(all_export[column]) else None,
                }
                for column in FEATURE_COLUMNS
            },
        },
        "split_summary": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_videos": len(split_map["train"]),
            "validation_videos": len(split_map["validation"]),
            "test_videos": len(split_map["test"]),
            "video_ids_by_split": split_map,
        },
        "data_leakage_checks": {
            "group_split_by_video_id": True,
            "same_video_in_multiple_splits": False,
            "congestion_score_excluded_from_model_features": True,
            "target_column_present_only_in_labeled_dataset": True,
        },
        "dataset_readiness": {
            "status": "insufficient_for_high_accuracy_generalized_model",
            "reason": "Only 38 aggregated time windows are available; this is a small dataset and should be treated as a starting point for a pipeline rather than a final production model.",
        },
    }
    return report


def _write_text_report(report: dict, split_map: dict[str, list[str]]) -> None:
    lines = [
        "Traffic Density Dataset Validation Report",
        "======================================",
        f"Total rows: {report['dataset_summary']['total_rows']}",
        f"Total videos: {report['dataset_summary']['total_videos']}",
        f"Total features: {report['dataset_summary']['total_features']}",
        f"Target classes: {', '.join(report['dataset_summary']['target_classes'])}",
        "",
        "Class distribution:",
    ]
    for label, count in report['dataset_summary']['class_distribution'].items():
        lines.append(f"- {label}: {count}")
    lines.extend([
        "",
        "Split summary:",
        f"- train rows: {report['split_summary']['train_rows']} | videos: {split_map['train']}",
        f"- validation rows: {report['split_summary']['validation_rows']} | videos: {split_map['validation']}",
        f"- test rows: {report['split_summary']['test_rows']} | videos: {split_map['test']}",
        "",
        "Data leakage checks:",
        "- Group split performed by video_id only.",
        "- Same video never appears in more than one split.",
        "- congestion_score intentionally excluded to avoid target leakage.",
        "- traffic_density is the only target variable.",
        "",
        "Dataset readiness:",
        "- The current dataset size is insufficient to claim a highly accurate generalized traffic-density model.",
        "- More videos should be added to data/videos/training/ and the pipeline rerun.",
    ])
    (FINAL_DIR / "dataset_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    raw_report = save_dataset_integration_report(RAW_DIR, RAW_DIR / "raw_dataset_integration_report.json")
    labeled_df = _build_labeled_dataset()
    train_df, val_df, test_df, split_map = _group_split(labeled_df)

    train_export = _export_split_frames(train_df)
    val_export = _export_split_frames(val_df)
    test_export = _export_split_frames(test_df)

    train_export.to_csv(FINAL_DIR / "train.csv", index=False)
    val_export.to_csv(FINAL_DIR / "validation.csv", index=False)
    test_export.to_csv(FINAL_DIR / "test.csv", index=False)

    report = _build_report(train_df, val_df, test_df, split_map)
    (FINAL_DIR / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_text_report(report, split_map)

    print("Raw dataset integration report:")
    print(json.dumps(raw_report, indent=2))
    print("\nFinal dataset summary:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
