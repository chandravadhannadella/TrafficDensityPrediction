from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from model.inference.model_config import NoOpScaler
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "datasets" / "final"
MODEL_DIR = ROOT / "model" / "saved_models"
EVAL_DIR = ROOT / "model" / "evaluation"
TARGET_COLUMN = "traffic_density"
LABEL_ORDER = ["LOW", "MEDIUM", "HIGH"]


def _load_split(split_name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}' in {path.name}")
    return df


def _feature_columns(columns: list[str] | pd.Index) -> list[str]:
    excluded = {
        TARGET_COLUMN,
        "avg_total_vehicle_count",
        "congestion_score",
        "video_id",
        "source_id",
        "window_start",
        "window_end",
        "video_name",
        "split",
        "label_source",
        "group_id",
    }
    features: list[str] = []
    for name in columns:
        value = str(name)
        if value in excluded:
            continue
        if value not in features:
            features.append(value)
    return features


def _load_dataset() -> tuple[pd.DataFrame, list[str], str | None]:
    frames: list[pd.DataFrame] = []
    for split_name in ["train", "validation", "test"]:
        split = _load_split(split_name).copy()
        split["split"] = split_name
        frames.append(split)
    combined = pd.concat(frames, ignore_index=True)
    feature_names = _feature_columns(combined.columns)
    group_column = "video_id" if "video_id" in combined.columns else "source_id" if "source_id" in combined.columns else None
    return combined, feature_names, group_column


def _make_model_specs() -> list[tuple[str, Any]]:
    return [
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=5000,
                random_state=42,
                class_weight="balanced",
                solver="lbfgs",
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                class_weight="balanced_subsample",
                min_samples_leaf=1,
                min_samples_split=2,
            ),
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(
                random_state=42,
                learning_rate=0.05,
                n_estimators=300,
                max_depth=3,
            ),
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=6,
                max_iter=300,
                random_state=42,
            ),
        ),
        (
            "SVM",
            SVC(
                C=2.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=LABEL_ORDER, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).tolist(),
    }


def _get_cv_iterator(data: pd.DataFrame, group_column: str | None) -> Any:
    y = data[TARGET_COLUMN].astype(str)
    if group_column is not None and group_column in data.columns and data[group_column].notna().any():
        groups = data[group_column].astype(str)
        n_groups = groups.nunique()
        if n_groups >= 3:
            return StratifiedGroupKFold(n_splits=min(3, n_groups)), groups
    if len(np.unique(y)) >= 2:
        return StratifiedKFold(n_splits=min(3, max(2, min(5, y.shape[0])))), None
    return None, None


def _select_model_for_training(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame | None = None) -> tuple[Any, Any]:
    if model_name in {"Logistic Regression", "SVM"}:
        scaler = StandardScaler()
        X_train_model = scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_model = scaler.transform(X_val)
        else:
            X_val_model = None
        if model_name == "Logistic Regression":
            estimator = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced", solver="lbfgs")
        else:
            estimator = SVC(C=2.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=42)
        return estimator, (scaler, X_train_model, X_val_model)
    return None, None


def _model_fit_predict(model_name: str, estimator: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, scaler: Any = None) -> np.ndarray:
    if model_name in {"Logistic Regression", "SVM"}:
        if scaler is None:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        estimator.fit(X_train, y_train)
        return estimator.predict(X_test)

    if model_name == "Random Forest":
        estimator.fit(X_train, y_train)
        return estimator.predict(X_test)

    if model_name == "Gradient Boosting":
        estimator.fit(X_train, y_train)
        return estimator.predict(X_test)

    if model_name == "HistGradientBoosting":
        estimator.fit(X_train, y_train)
        return estimator.predict(X_test)

    estimator.fit(X_train, y_train)
    return estimator.predict(X_test)


def _group_cv_scores(data: pd.DataFrame, feature_names: list[str], model_name: str, group_column: str | None) -> tuple[float, float, list[float], list[float]]:
    X = data[feature_names]
    y = data[TARGET_COLUMN].astype(str)
    cv_iterator, groups = _get_cv_iterator(data, group_column)
    if cv_iterator is None:
        return 0.0, 0.0, [], []

    accuracy_scores: list[float] = []
    macro_scores: list[float] = []
    for split_index, (train_idx, test_idx) in enumerate(cv_iterator.split(X, y, groups) if groups is not None else cv_iterator.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if model_name in {"Logistic Regression", "SVM"}:
            scaler = StandardScaler()
            X_train_model = scaler.fit_transform(X_train)
            X_test_model = scaler.transform(X_test)
            if model_name == "Logistic Regression":
                estimator = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")
            else:
                estimator = SVC(C=2.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=42)
            estimator.fit(X_train_model, y_train)
            pred = estimator.predict(X_test_model)
        elif model_name == "Random Forest":
            estimator = RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                class_weight="balanced_subsample",
                min_samples_leaf=1,
                min_samples_split=2,
            )
            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_test)
        elif model_name == "Gradient Boosting":
            estimator = GradientBoostingClassifier(
                random_state=42,
                learning_rate=0.05,
                n_estimators=300,
                max_depth=3,
            )
            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_test)
        else:
            estimator = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=6,
                max_iter=300,
                random_state=42,
            )
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


def _build_feature_importance(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        return [{"feature": feature, "importance": float(importance)} for feature, importance in zip(feature_names, importances, strict=True)]
    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)
        importances = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
        return [{"feature": feature, "importance": float(importance)} for feature, importance in zip(feature_names, importances, strict=True)]
    return [{"feature": feature, "importance": 0.0} for feature in feature_names]


def train_models_v3(output_dir: str | Path | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir) if output_dir is not None else MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_split("train").copy()
    val_df = _load_split("validation").copy()
    test_df = _load_split("test").copy()
    combined_df, feature_names, group_column = _load_dataset()

    if not feature_names:
        raise ValueError("No valid feature columns were found after excluding target and leakage columns.")

    X_train = train_df[feature_names]
    X_val = val_df[feature_names]
    X_test = test_df[feature_names]
    y_train = train_df[TARGET_COLUMN].astype(str)
    y_val = val_df[TARGET_COLUMN].astype(str)
    y_test = test_df[TARGET_COLUMN].astype(str)

    model_results: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}
    feature_importance_rows: list[dict[str, Any]] = []

    for model_name, estimator in _make_model_specs():
        scaler = StandardScaler() if model_name in {"Logistic Regression", "SVM"} else None
        if scaler is not None:
            X_train_model = scaler.fit_transform(X_train)
            X_val_model = scaler.transform(X_val)
            X_test_model = scaler.transform(X_test)
            model = clone(estimator)
            model.fit(X_train_model, y_train)
            val_pred = model.predict(X_val_model)
            test_pred = model.predict(X_test_model)
        else:
            model = clone(estimator)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

        validation_metrics = _metrics(y_val, val_pred)
        test_metrics = _metrics(y_test, test_pred)
        cv_acc_mean, cv_macro_mean, cv_acc_scores, cv_macro_scores = _group_cv_scores(combined_df, feature_names, model_name, group_column)

        result_row = {
            "model_name": model_name,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "group_cv_accuracy_mean": cv_acc_mean,
            "group_cv_macro_f1_mean": cv_macro_mean,
            "group_cv_accuracy_scores": cv_acc_scores,
            "group_cv_macro_f1_scores": cv_macro_scores,
            "feature_importance": _build_feature_importance(model, feature_names),
        }
        model_results.append(result_row)
        trained_models[model_name] = model
        feature_importance_rows.extend(
            {"model_name": model_name, "feature": item["feature"], "importance": item["importance"]}
            for item in result_row["feature_importance"]
        )

    comparison_rows: list[dict[str, Any]] = []
    for row in model_results:
        comparison_rows.append(
            {
                "model_name": row["model_name"],
                "validation_accuracy": row["validation_metrics"]["accuracy"],
                "validation_macro_f1": row["validation_metrics"]["macro_f1"],
                "validation_weighted_f1": row["validation_metrics"]["weighted_f1"],
                "test_accuracy": row["test_metrics"]["accuracy"],
                "test_macro_f1": row["test_metrics"]["macro_f1"],
                "test_weighted_f1": row["test_metrics"]["weighted_f1"],
                "group_cv_accuracy_mean": row["group_cv_accuracy_mean"],
                "group_cv_macro_f1_mean": row["group_cv_macro_f1_mean"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    ranked = comparison_df.sort_values(
        ["test_macro_f1", "validation_macro_f1", "group_cv_macro_f1_mean", "test_accuracy"],
        ascending=False,
    )
    best_model_name = str(ranked.iloc[0]["model_name"])
    best_model = trained_models[best_model_name]
    if best_model_name in {"Logistic Regression", "SVM"}:
        best_scaler = StandardScaler()
        best_scaler.fit(X_train)
    else:
        best_scaler = NoOpScaler()
        best_scaler.fit(X_train)

    comparison_df.to_csv(output_dir / "model_comparison_v3.csv", index=False)
    pd.DataFrame(feature_importance_rows).to_csv(output_dir / "feature_importance_v3.csv", index=False)

    with (output_dir / "feature_names_v3.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)

    with (output_dir / "scaler_v3.pkl").open("wb") as handle:
        pickle.dump(best_scaler, handle)

    with (output_dir / "traffic_density_model_v3.pkl").open("wb") as handle:
        pickle.dump(best_model, handle)

    metadata = {
        "model_name": best_model_name,
        "model_version": "v3",
        "training_dataset_size": int(len(train_df)),
        "image_sample_count": int(train_df.shape[0]),
        "video_sample_count": int(0),
        "class_distribution": {label: int(pd.concat([train_df, val_df, test_df], ignore_index=True)[TARGET_COLUMN].astype(str).eq(label).sum()) for label in LABEL_ORDER},
        "feature_names": feature_names,
        "preprocessing_method": "StandardScaler" if isinstance(best_scaler, StandardScaler) else "NoOpScaler",
        "training_date": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "validation_accuracy": float(ranked.iloc[0]["validation_accuracy"]),
        "test_accuracy": float(ranked.iloc[0]["test_accuracy"]),
        "validation_macro_f1": float(ranked.iloc[0]["validation_macro_f1"]),
        "test_macro_f1": float(ranked.iloc[0]["test_macro_f1"]),
        "cross_validation_results": {
            "accuracy_mean": float(ranked.iloc[0]["group_cv_accuracy_mean"]),
            "macro_f1_mean": float(ranked.iloc[0]["group_cv_macro_f1_mean"]),
        },
        "label_source": "proxy_thresholds",
        "training_configuration": {
            "split_strategy": "group-aware when video_id/group_id available else stratified fallback",
            "label_order": LABEL_ORDER,
            "models_tested": [row["model_name"] for row in model_results],
        },
    }
    with (output_dir / "model_metadata_v3.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "feature_names": feature_names,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "scaler": best_scaler,
        "metadata": metadata,
        "comparison_df": comparison_df,
        "model_results": model_results,
    }


def main() -> None:
    result = train_models_v3()
    print(result["best_model_name"])
    print(result["comparison_df"].to_string(index=False))


if __name__ == "__main__":
    main()
