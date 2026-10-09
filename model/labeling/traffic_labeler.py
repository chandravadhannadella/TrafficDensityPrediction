"""Traffic density labeling utilities.

This module intentionally separates three label sources:

1. Existing labeled datasets already present in the project or external CSVs.
2. Manually reviewed labels supplied by a reviewer for specific video windows.
3. A documented fallback heuristic based on average vehicle count thresholds.

The final fallback labels are proxy labels only and must not be confused with
human ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DEFAULT_LOW_THRESHOLD = 4.0
DEFAULT_MEDIUM_THRESHOLD = 10.0
LABEL_ORDER = ["LOW", "MEDIUM", "HIGH"]


def _normalize_name(name: Any) -> str:
    if name is None:
        return ""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _column_aliases() -> dict[str, list[str]]:
    return {
        "traffic_density": [
            "traffic_density",
            "density_label",
            "traffic_level",
            "density",
            "density_level",
            "label",
            "segment_density",
            "road_density",
        ],
        "video_id": ["video_id", "video", "video_name"],
        "window_start": ["window_start", "time_window_start", "start_time", "timestamp_start"],
        "window_end": ["window_end", "time_window_end", "end_time", "timestamp_end"],
        "avg_total_vehicle_count": [
            "avg_total_vehicle_count",
            "average_total_vehicle_count",
            "total_vehicle_count",
            "vehicle_count",
            "avg_vehicles",
            "traffic_volume",
            "vehicle_volume",
        ],
        "traffic_flow": ["traffic_flow", "flow_rate", "flow"],
        "congestion_score": ["congestion_score", "congestion", "traffic_congestion"],
        "average_speed": ["average_speed", "avg_speed", "speed"],
        "road_occupancy": ["road_occupancy", "occupancy", "road_usage"],
        "weather": ["weather", "weather_condition"],
        "time_of_day": ["time_of_day", "hour", "time"],
        "date": ["date", "day"],
    }


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a dataframe schema to the project's standard names."""
    normalized = df.copy()
    aliases = _column_aliases()
    for target, names in aliases.items():
        for candidate in names:
            matching = [column for column in normalized.columns if _normalize_name(column) == candidate]
            if matching:
                normalized = normalized.rename(columns={matching[0]: target})
                break
    return normalized


def detect_existing_label_column(df: pd.DataFrame) -> str | None:
    """Return the first valid traffic-density label column discovered in a dataset."""
    normalized = normalize_dataframe_columns(df)
    aliases = _column_aliases()["traffic_density"]
    for alias in aliases:
        if alias in normalized.columns:
            return alias
    for column in normalized.columns:
        if _normalize_name(column).endswith("traffic_density") or _normalize_name(column).endswith("density"):
            return column
    return None


def assign_threshold_labels(
    values: Iterable[float] | pd.Series,
    *,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
) -> pd.Series:
    """Assign LOW/MEDIUM/HIGH labels using domain thresholds.

    These are proxy labels derived from vehicle counts and are not human ground
    truth. They are stored separately from any manually reviewed labels.
    """
    series = pd.Series(values, copy=True)
    labels = pd.Series("LOW", index=series.index, dtype=object)
    labels[series >= medium_threshold] = "HIGH"
    labels[(series >= low_threshold) & (series < medium_threshold)] = "MEDIUM"
    return labels


class TrafficLabeler:
    """Prepare traffic-density labels using the best available source."""

    def __init__(
        self,
        *,
        low_threshold: float = DEFAULT_LOW_THRESHOLD,
        medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
        source_feature: str = "avg_total_vehicle_count",
        target_column: str = "traffic_density",
    ) -> None:
        self.low_threshold = float(low_threshold)
        self.medium_threshold = float(medium_threshold)
        self.source_feature = source_feature
        self.target_column = target_column

    def _ensure_standard_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        return normalize_dataframe_columns(df)

    def _existing_label_series(self, df: pd.DataFrame) -> pd.Series | None:
        normalized = self._ensure_standard_schema(df)
        candidate_cols = [
            "traffic_density",
            "density_label",
            "traffic_level",
            "density",
            "label",
        ]
        for col in candidate_cols:
            if col in normalized.columns:
                series = normalized[col].copy()
                return series.map(lambda x: str(x).strip().upper() if pd.notna(x) else x)
        for col in normalized.columns:
            if _normalize_name(col).endswith("traffic_density") or _normalize_name(col).endswith("density"):
                series = normalized[col].copy()
                return series.map(lambda x: str(x).strip().upper() if pd.notna(x) else x)
        return None

    def assign_proxy_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply proxy labels using average vehicle count thresholds.

        This is a fallback only when no hand-labeled data is available.
        """
        result = df.copy()
        result = self._ensure_standard_schema(result)
        source = self.source_feature if self.source_feature in result.columns else None
        if source is None:
            for candidate in ["avg_total_vehicle_count", "total_vehicle_count", "vehicle_count", "traffic_volume"]:
                if candidate in result.columns:
                    source = candidate
                    break
        if source is None:
            raise ValueError(f"No compatible count feature found for proxy labeling. Expected one of {self.source_feature}, avg_total_vehicle_count, total_vehicle_count, vehicle_count, traffic_volume.")

        labels = assign_threshold_labels(
            result[source],
            low_threshold=self.low_threshold,
            medium_threshold=self.medium_threshold,
        )
        result[self.target_column] = labels
        return result

    def apply_manual_review_labels(self, df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
        """Merge in a manually reviewed label table if available."""
        result = self._ensure_standard_schema(df).copy()
        manual = self._ensure_standard_schema(manual_df).copy()

        join_keys = []
        for key in ["video_id", "window_start", "window_end"]:
            if key in result.columns and key in manual.columns:
                join_keys.append(key)
        if not join_keys:
            raise ValueError("Manual review table does not contain a compatible join key (video_id/window_start/window_end).")

        label_col = detect_existing_label_column(manual)
        if label_col is None:
            raise ValueError("Manual review table is missing a traffic-density label column.")

        manual_relevant = manual[[*join_keys, label_col]].copy()
        manual_relevant[label_col] = manual_relevant[label_col].map(lambda x: str(x).strip().upper() if pd.notna(x) else x)

        def _key_from_row(row: pd.Series) -> tuple[str, ...]:
            return tuple(str(value).strip() for value in row.tolist())

        lookup = {
            _key_from_row(manual_relevant.loc[idx, join_keys]): str(manual_relevant.loc[idx, label_col])
            for idx in manual_relevant.index
        }

        result[self.target_column] = result[join_keys].apply(_key_from_row, axis=1).map(lambda key: lookup.get(key, pd.NA))
        result[self.target_column] = result[self.target_column].map(lambda x: str(x).strip().upper() if pd.notna(x) else x)
        return result

    def label_dataframe(self, df: pd.DataFrame, manual_labels: pd.DataFrame | None = None) -> pd.DataFrame:
        """Apply the best available label source in the following priority:

        1. Existing labeled traffic datasets
        2. Manually reviewed window labels
        3. Proxy labels from average vehicle count thresholds
        """
        result = self._ensure_standard_schema(df)
        existing = self._existing_label_series(result)
        if existing is not None:
            result[self.target_column] = existing.map(lambda x: str(x).strip().upper() if pd.notna(x) else x)
            return result
        if manual_labels is not None:
            result = self.apply_manual_review_labels(result, manual_labels)
            if self.target_column in result.columns:
                return result
        return self.assign_proxy_labels(result)


def detect_compatible_csv_datasets(raw_dir: str | Path) -> list[dict[str, Any]]:
    """Inspect raw CSV files and report whether they are useful for traffic-density training.

    Only datasets with actual traffic-density labels or obvious compatible features are
    accepted for integration. Unrelated or empty CSV files are ignored.
    """
    raw_root = Path(raw_dir)
    if not raw_root.exists():
        return []

    infos: list[dict[str, Any]] = []
    for csv_path in sorted(raw_root.glob("*.csv")):
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        if frame.empty:
            infos.append({"file": str(csv_path.name), "status": "empty", "reason": "CSV is empty"})
            continue
        normalized = normalize_dataframe_columns(frame)
        score = 0
        for candidate in [
            "traffic_density",
            "traffic_level",
            "density",
            "congestion_score",
            "congestion",
            "vehicle_count",
            "traffic_volume",
            "avg_total_vehicle_count",
            "road_occupancy",
            "weather",
            "time_of_day",
            "date",
        ]:
            if candidate in normalized.columns:
                score += 1
        label_col = detect_existing_label_column(frame)
        if label_col is not None:
            score += 5

        if score >= 3:
            infos.append(
                {
                    "file": str(csv_path.name),
                    "status": "compatible",
                    "columns": list(frame.columns),
                    "normalized_columns": list(normalized.columns),
                    "has_density_label": label_col is not None,
                    "label_column": label_col,
                }
            )
        else:
            infos.append({
                "file": str(csv_path.name),
                "status": "not_compatible",
                "columns": list(frame.columns),
                "reason": "No actual density label or compatible traffic features found.",
            })
    return infos


def save_dataset_integration_report(raw_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Create a JSON report describing available raw CSV integration candidates."""
    report = {
        "raw_dataset_directory": str(Path(raw_dir)),
        "compatible_datasets": detect_compatible_csv_datasets(raw_dir),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = save_dataset_integration_report("data/datasets/raw", "data/datasets/raw/raw_dataset_integration_report.json")
    print(json.dumps(report, indent=2))
