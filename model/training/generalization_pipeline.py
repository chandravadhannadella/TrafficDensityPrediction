from __future__ import annotations

import math
import random
from typing import Any

import pandas as pd


def _normalize_label_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().upper() if pd.notna(value) else value)


def build_grouped_split(
    df: pd.DataFrame,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split a dataframe by video_id to avoid leakage across traffic videos."""
    if not {"video_id"}.issubset(df.columns):
        raise ValueError("Grouped split requires a 'video_id' column.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isnan(total_ratio) and not math.isclose(total_ratio, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    unique_videos = sorted(df["video_id"].dropna().astype(str).unique().tolist())
    if len(unique_videos) < 3:
        raise ValueError("A grouped split needs at least 3 unique videos.")

    rng = random.Random(random_seed)
    shuffled = unique_videos[:]
    rng.shuffle(shuffled)

    if train_ratio > 0:
        train_count = max(1, round(len(shuffled) * train_ratio))
    else:
        train_count = 0
    if val_ratio > 0:
        val_count = max(1, round(len(shuffled) * val_ratio))
    else:
        val_count = 0
    test_count = len(shuffled) - train_count - val_count
    if test_count < 1:
        test_count = 1
        if train_count + val_count + test_count > len(shuffled):
            train_count = max(1, len(shuffled) // 2)
            val_count = max(1, (len(shuffled) - train_count) // 2)
            test_count = len(shuffled) - train_count - val_count

    train_videos = shuffled[:train_count]
    val_videos = shuffled[train_count:train_count + val_count]
    test_videos = shuffled[train_count + val_count:train_count + val_count + test_count]

    split_map = {
        "train": df[df["video_id"].astype(str).isin(train_videos)].copy().reset_index(drop=True),
        "validation": df[df["video_id"].astype(str).isin(val_videos)].copy().reset_index(drop=True),
        "test": df[df["video_id"].astype(str).isin(test_videos)].copy().reset_index(drop=True),
    }
    return split_map


def summarize_label_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize label availability and whether the dataset is proxy or human-reviewed."""
    if "traffic_density" not in df.columns:
        return {
            "label_source": "missing",
            "ground_truth_status": "unknown",
            "class_distribution": {},
            "missing_labels": int(df.shape[0]),
        }

    labels = _normalize_label_series(df["traffic_density"])
    counts = labels.value_counts().sort_index().to_dict()
    non_missing = {str(key): int(value) for key, value in counts.items()}

    label_source = "proxy_thresholds"
    if "label_source" in df.columns:
        sources = _normalize_label_series(df["label_source"]).astype(str)
        if not sources.empty:
            source_values = set(sources.dropna().unique().tolist())
            if "MANUAL" in source_values or "MANUAL_REVIEW" in source_values:
                label_source = "manual_review"
    ground_truth_status = "not_verified"
    if label_source == "manual_review":
        ground_truth_status = "reviewed_but_unverified"

    return {
        "label_source": label_source,
        "ground_truth_status": ground_truth_status,
        "class_distribution": non_missing,
        "missing_labels": int(labels.isna().sum()),
    }
