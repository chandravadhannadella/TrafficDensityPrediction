from __future__ import annotations

import pandas as pd

from model.labeling.traffic_labeler import TrafficLabeler
from model.training.generalization_pipeline import build_grouped_split, summarize_label_quality


def _make_window_frame() -> pd.DataFrame:
    rows = []
    for video_id in ["v1", "v2", "v3", "v4", "v5", "v6"]:
        for window_start in [0, 10]:
            rows.append(
                {
                    "video_id": video_id,
                    "video_name": f"{video_id}.mp4",
                    "window_start": window_start,
                    "window_end": window_start + 10,
                    "avg_total_vehicle_count": 1.0 if video_id in {"v1", "v2"} else 6.0 if video_id in {"v3", "v4"} else 12.0,
                    "traffic_flow": 2.0,
                    "congestion_score": 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_label_dataframe_prefers_manual_review_over_proxy_labels() -> None:
    df = pd.DataFrame(
        [
            {
                "video_id": "v1",
                "video_name": "v1.mp4",
                "window_start": 0,
                "window_end": 10,
                "avg_total_vehicle_count": 1.0,
                "traffic_flow": 2.0,
                "congestion_score": 0.1,
            },
            {
                "video_id": "v2",
                "video_name": "v2.mp4",
                "window_start": 10,
                "window_end": 20,
                "avg_total_vehicle_count": 12.0,
                "traffic_flow": 2.0,
                "congestion_score": 0.1,
            },
        ]
    )
    manual = pd.DataFrame(
        {
            "video_id": ["v1", "v2"],
            "window_start": [0, 10],
            "window_end": [10, 20],
            "traffic_density": ["HIGH", "LOW"],
        }
    )

    result = TrafficLabeler().label_dataframe(df, manual_labels=manual)

    assert set(result["traffic_density"]) == {"HIGH", "LOW"}
    assert result.loc[result["video_id"] == "v1", "traffic_density"].iloc[0] == "HIGH"
    assert result.loc[result["video_id"] == "v2", "traffic_density"].iloc[0] == "LOW"


def test_build_grouped_split_keeps_videos_out_of_multiple_splits() -> None:
    df = _make_window_frame()

    split = build_grouped_split(df, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, random_seed=42)

    train_ids = split["train"]["video_id"].unique().tolist()
    val_ids = split["validation"]["video_id"].unique().tolist()
    test_ids = split["test"]["video_id"].unique().tolist()

    assert set(train_ids).isdisjoint(set(val_ids))
    assert set(train_ids).isdisjoint(set(test_ids))
    assert set(val_ids).isdisjoint(set(test_ids))
    assert len(train_ids) + len(val_ids) + len(test_ids) == df["video_id"].nunique()


def test_summarize_label_quality_flags_proxy_only_datasets() -> None:
    df = _make_window_frame().copy()
    df["traffic_density"] = ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH", "HIGH", "HIGH", "HIGH", "HIGH", "HIGH", "HIGH"]

    report = summarize_label_quality(df)

    assert report["class_distribution"]["LOW"] >= 1
    assert report["label_source"] == "proxy_thresholds"
    assert report["ground_truth_status"] == "not_verified"
