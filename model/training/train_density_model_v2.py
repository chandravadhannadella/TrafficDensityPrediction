from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "datasets" / "final"
MODEL_DIR = ROOT / "model" / "saved_models"
LABEL_ORDER = ["LOW", "MEDIUM", "HIGH"]
TARGET_COLUMN = "traffic_density"


def _load_split(split_name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}' in {path.name}")
    return df


def _feature_columns(columns: list[str] | pd.Index) -> list[str]:
    excluded = {TARGET_COLUMN, "avg_total_vehicle_count", "congestion_score", "video_id", "window_start", "window_end", "video_name", "split"}
    features = []
    for name in columns:
        value = str(name)
        if value in excluded:
            continue
        if value not in features:
            features.append(value)
    return features


def _load_dataset() -> tuple[pd.DataFrame, list[str]]:
    processed_path = ROOT / "data" / "datasets" / "processed" / "labeled_traffic_features.csv"
    if processed_path.exists():
        combined = pd.read_csv(processed_path)
        combined = combined.copy()
        feature_names = _feature_columns(pd.read_csv(DATA_DIR / "train.csv").columns)
        return combined, feature_names

    frames = []
    for split_name in ["train", "validation", "test"]:
        split = _load_split(split_name)
        split = split.copy()
        split["split"] = split_name
        frames.append(split)
    combined = pd.concat(frames, ignore_index=True)
    features = _feature_columns(combined.columns)
    return combined, features


def _make_model_specs() -> list[tuple[str, object]]:
    return [
        ("Logistic Regression", LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")),
        ("Random Forest", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced_subsample")),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
    ]


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    labels = LABEL_ORDER
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _group_cv_scores(data: pd.DataFrame, feature_names: list[str], model_name: str) -> tuple[float, float, list[float], list[float]]:
    X = data[feature_names]
    y = data[TARGET_COLUMN].astype(str)
    if "video_id" in data.columns:
        groups = data["video_id"]
        n_splits = min(3, data["video_id"].nunique())
    else:
        groups = np.arange(len(data))
        n_splits = min(3, len(data))
    cv = GroupKFold(n_splits=max(2, min(3, n_splits)))
    accuracy_scores: list[float] = []
    macro_scores: list[float] = []

    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if model_name == "Logistic Regression":
            scaler = StandardScaler()
            X_train_model = scaler.fit_transform(X_train)
            X_test_model = scaler.transform(X_test)
            estimator = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")
            estimator.fit(X_train_model, y_train)
            pred = estimator.predict(X_test_model)
        elif model_name == "Random Forest":
            estimator = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced_subsample")
            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_test)
        else:
            estimator = GradientBoostingClassifier(random_state=42)
            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_test)

        accuracy_scores.append(float(accuracy_score(y_test, pred)))
        macro_scores.append(float(f1_score(y_test, pred, average="macro", labels=LABEL_ORDER, zero_division=0)))

    return (
        float(np.mean(accuracy_scores)) if accuracy_scores else 0.0,
        float(np.mean(macro_scores)) if macro_scores else 0.0,
        accuracy_scores,
        macro_scores,
    )


def train_models_v2(output_dir: str | Path | None = None) -> dict[str, object]:
    output_dir = Path(output_dir) if output_dir is not None else MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_split("train").copy()
    val_df = _load_split("validation").copy()
    test_df = _load_split("test").copy()
    combined_df, feature_names = _load_dataset()
    combined_df = combined_df[[*feature_names, TARGET_COLUMN, "video_id"]].copy()

    X_train = train_df[feature_names]
    X_val = val_df[feature_names]
    X_test = test_df[feature_names]
    y_train = train_df[TARGET_COLUMN].astype(str)
    y_val = val_df[TARGET_COLUMN].astype(str)
    y_test = test_df[TARGET_COLUMN].astype(str)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model_results: list[dict[str, object]] = []
    trained: dict[str, object] = {}

    for model_name, estimator in _make_model_specs():
        if model_name == "Logistic Regression":
            model = estimator
            model.fit(X_train_scaled, y_train)
            val_pred = model.predict(X_val_scaled)
            test_pred = model.predict(X_test_scaled)
        else:
            model = estimator
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

        metrics = {
            "validation": _metrics(y_val, val_pred),
            "test": _metrics(y_test, test_pred),
        }

        best_cv_acc, best_cv_macro, cv_accs, cv_macro_scores = _group_cv_scores(combined_df, feature_names, model_name)
        row = {
            "model_name": model_name,
            "validation_metrics": metrics["validation"],
            "test_metrics": metrics["test"],
            "group_cv_accuracy_mean": best_cv_acc,
            "group_cv_macro_f1_mean": best_cv_macro,
            "group_cv_accuracy_scores": cv_accs,
            "group_cv_macro_f1_scores": cv_macro_scores,
        }
        model_results.append(row)
        trained[model_name] = model

    comparison_rows = []
    for row in model_results:
        comparison_rows.append(
            {
                "model_name": row["model_name"],
                "validation_accuracy": row["validation_metrics"]["accuracy"],
                "validation_macro_f1": row["validation_metrics"]["macro_f1"],
                "test_accuracy": row["test_metrics"]["accuracy"],
                "test_macro_f1": row["test_metrics"]["macro_f1"],
                "cv_accuracy_mean": row["group_cv_accuracy_mean"],
                "cv_macro_f1_mean": row["group_cv_macro_f1_mean"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    ranked = comparison_df.sort_values(["validation_macro_f1", "test_macro_f1", "cv_macro_f1_mean"], ascending=False)
    best_model_name = ranked.iloc[0]["model_name"]
    best_model = trained[best_model_name]

    with (output_dir / "feature_names_v2.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)

    with (output_dir / "scaler_v2.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)

    with (output_dir / "traffic_density_model_v2.pkl").open("wb") as handle:
        pickle.dump(best_model, handle)

    metadata = {
        "target_column": TARGET_COLUMN,
        "label_order": LABEL_ORDER,
        "feature_names": feature_names,
        "best_model_name": best_model_name,
        "proxy_label_note": "Labels remain proxy-based thresholds derived from average vehicle counts; they are not manually verified ground truth.",
        "data_files": {
            "train": str(DATA_DIR / "train.csv"),
            "validation": str(DATA_DIR / "validation.csv"),
            "test": str(DATA_DIR / "test.csv"),
        },
        "validation_metrics": {
            best_model_name: {
                "accuracy": float(ranked.iloc[0]["validation_accuracy"]),
                "macro_f1": float(ranked.iloc[0]["validation_macro_f1"]),
            }
        },
        "test_metrics": {
            best_model_name: {
                "accuracy": float(ranked.iloc[0]["test_accuracy"]),
                "macro_f1": float(ranked.iloc[0]["test_macro_f1"]),
            }
        },
        "group_cv_accuracy_mean": float(ranked.iloc[0]["cv_accuracy_mean"]),
        "group_cv_macro_f1_mean": float(ranked.iloc[0]["cv_macro_f1_mean"]),
    }
    with (output_dir / "model_metadata_v2.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "feature_names": feature_names,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "scaler": scaler,
        "metadata": metadata,
        "comparison_df": comparison_df,
    }


def main() -> None:
    out = train_models_v2()
    print(out["best_model_name"])
    print(out["comparison_df"].to_string(index=False))


if __name__ == "__main__":
    main()
