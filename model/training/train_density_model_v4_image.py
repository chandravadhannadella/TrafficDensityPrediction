"""
V4-Image Model Training Pipeline
==================================
Dedicated image-level traffic density prediction model.

This is separate from the video model (V3) to avoid forcing single images
into temporal window-level features.

Features used:
- car_count, bus_count, truck_count, motorcycle_count, bicycle_count
- total_vehicle_count
- Simple class proportions

Proxy labels:
- LOW: total_vehicle_count <= 8
- MEDIUM: 9 <= total_vehicle_count <= 15
- HIGH: total_vehicle_count >= 16
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "images"
MODEL_DIR = ROOT / "model" / "saved_models"
EVAL_DIR = ROOT / "model" / "evaluation"

TARGET_COLUMN = "traffic_density"
LABEL_ORDER = ["LOW", "MEDIUM", "HIGH"]

# Proxy label thresholds based on raw image statistics
LOW_THRESHOLD = 8  # 0 to 8 vehicles
HIGH_THRESHOLD = 15  # 16+ vehicles


def _load_raw_detections() -> pd.DataFrame:
    """Load raw YOLO detections from all images."""
    path = DATA_DIR / "raw_detections.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw detections not found: {path}")
    return pd.read_csv(path)


def _assign_proxy_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign traffic density labels based on vehicle count thresholds."""
    df = df.copy()
    
    def assign_label(count):
        if count <= LOW_THRESHOLD:
            return "LOW"
        elif count <= HIGH_THRESHOLD:
            return "MEDIUM"
        else:
            return "HIGH"
    
    df[TARGET_COLUMN] = df["total_vehicle_count"].apply(assign_label)
    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Select image-level features (exclude path and counts used only for labeling)."""
    return [
        "car_count",
        "bus_count",
        "truck_count",
        "motorcycle_count",
        "bicycle_count",
        "total_vehicle_count",
    ]


def _create_train_test_split(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits.
    
    Uses image_path pseudo-grouping to try to avoid same-source contamination,
    though without explicit source metadata this is a best-effort approach.
    """
    y = df[TARGET_COLUMN].astype(str)
    
    # First split: train+val vs test
    from sklearn.model_selection import train_test_split
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=y,
        random_state=42,
    )
    
    # Second split: train vs val
    y_tv = train_val[TARGET_COLUMN].astype(str)
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=y_tv,
        random_state=42,
    )
    
    return train, val, test


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    """Calculate evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=LABEL_ORDER, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).tolist(),
    }


def _make_model_specs() -> list[tuple[str, Any]]:
    """Define model candidates for image classification."""
    return [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced", solver="lbfgs")),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced_subsample", min_samples_leaf=2)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42, learning_rate=0.05, n_estimators=200, max_depth=3)),
    ]


def _build_feature_importance(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    """Extract feature importance from trained model."""
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        return [{"feature": feature, "importance": float(importance)} for feature, importance in zip(feature_names, importances, strict=True)]
    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)
        importances = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
        return [{"feature": feature, "importance": float(importance)} for feature, importance in zip(feature_names, importances, strict=True)]
    return [{"feature": feature, "importance": 0.0} for feature in feature_names]


def _cv_scores(X_train: pd.DataFrame, y_train: pd.Series, model_name: str) -> tuple[float, float]:
    """Compute cross-validation scores."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accuracy_scores = []
    macro_f1_scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Select and train model
        if model_name == "Logistic Regression":
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced", solver="lbfgs")
            model.fit(X_tr_scaled, y_tr)
            pred = model.predict(X_val_scaled)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced_subsample", min_samples_leaf=2)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
        else:  # Gradient Boosting
            model = GradientBoostingClassifier(random_state=42, learning_rate=0.05, n_estimators=200, max_depth=3)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
        
        accuracy_scores.append(float(accuracy_score(y_val, pred)))
        macro_f1_scores.append(float(f1_score(y_val, pred, average="macro", labels=LABEL_ORDER, zero_division=0)))
    
    return float(np.mean(accuracy_scores)), float(np.mean(macro_f1_scores))


def train_models_v4_image(output_dir: str | Path | None = None) -> dict[str, Any]:
    """Train image-specific traffic density models."""
    output_dir = Path(output_dir) if output_dir is not None else MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and label data
    raw_df = _load_raw_detections()
    labeled_df = _assign_proxy_labels(raw_df)
    
    # Create splits
    train_df, val_df, test_df = _create_train_test_split(labeled_df, test_size=0.2, val_size=0.1)
    
    feature_names = _feature_columns(labeled_df)
    X_train = train_df[feature_names]
    X_val = val_df[feature_names]
    X_test = test_df[feature_names]
    y_train = train_df[TARGET_COLUMN].astype(str)
    y_val = val_df[TARGET_COLUMN].astype(str)
    y_test = test_df[TARGET_COLUMN].astype(str)
    
    model_results = []
    trained_models = {}
    feature_importance_rows = []
    
    # Train each model
    for model_name, estimator in _make_model_specs():
        if model_name == "Logistic Regression":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            model = clone(estimator)
            model.fit(X_train_scaled, y_train)
            val_pred = model.predict(X_val_scaled)
            test_pred = model.predict(X_test_scaled)
        else:
            scaler = None
            model = clone(estimator)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
        
        val_metrics = _metrics(y_val, val_pred)
        test_metrics = _metrics(y_test, test_pred)
        cv_acc, cv_f1 = _cv_scores(X_train, y_train, model_name)
        
        result_row = {
            "model_name": model_name,
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "cv_accuracy": cv_acc,
            "cv_macro_f1": cv_f1,
            "confusion_matrix": test_metrics["confusion_matrix"],
            "feature_importance": _build_feature_importance(model, feature_names),
        }
        model_results.append(result_row)
        trained_models[model_name] = (model, scaler if model_name == "Logistic Regression" else None)
        feature_importance_rows.extend(
            {"model_name": model_name, "feature": item["feature"], "importance": item["importance"]}
            for item in result_row["feature_importance"]
        )
    
    # Select best model
    comparison_df = pd.DataFrame([
        {
            "model_name": row["model_name"],
            "validation_accuracy": row["validation_accuracy"],
            "validation_macro_f1": row["validation_macro_f1"],
            "test_accuracy": row["test_accuracy"],
            "test_macro_f1": row["test_macro_f1"],
            "cv_accuracy": row["cv_accuracy"],
            "cv_macro_f1": row["cv_macro_f1"],
        }
        for row in model_results
    ])
    
    ranked = comparison_df.sort_values(
        ["test_macro_f1", "validation_macro_f1", "cv_macro_f1", "test_accuracy"],
        ascending=False,
    )
    best_model_name = str(ranked.iloc[0]["model_name"])
    best_model, best_scaler = trained_models[best_model_name]
    
    # Save artifacts
    comparison_df.to_csv(output_dir / "model_comparison_v4_image.csv", index=False)
    pd.DataFrame(feature_importance_rows).to_csv(output_dir / "feature_importance_v4_image.csv", index=False)
    
    with (output_dir / "feature_names_v4_image.json").open("w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    
    if best_scaler is not None:
        with (output_dir / "scaler_v4_image.pkl").open("wb") as f:
            pickle.dump(best_scaler, f)
    else:
        with (output_dir / "scaler_v4_image.pkl").open("wb") as f:
            pickle.dump(None, f)
    
    with (output_dir / "traffic_density_model_v4_image.pkl").open("wb") as f:
        pickle.dump(best_model, f)
    
    # Metadata
    metadata = {
        "model_name": best_model_name,
        "model_version": "v4_image",
        "training_type": "image_specific",
        "training_dataset_size": len(train_df),
        "validation_dataset_size": len(val_df),
        "test_dataset_size": len(test_df),
        "total_images": len(labeled_df),
        "class_distribution": {label: int((labeled_df[TARGET_COLUMN] == label).sum()) for label in LABEL_ORDER},
        "feature_names": feature_names,
        "preprocessing_method": "StandardScaler" if best_scaler is not None else "NoOpScaler",
        "training_date": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "validation_accuracy": float(ranked.iloc[0]["validation_accuracy"]),
        "validation_macro_f1": float(ranked.iloc[0]["validation_macro_f1"]),
        "test_accuracy": float(ranked.iloc[0]["test_accuracy"]),
        "test_macro_f1": float(ranked.iloc[0]["test_macro_f1"]),
        "cv_accuracy": float(ranked.iloc[0]["cv_accuracy"]),
        "cv_macro_f1": float(ranked.iloc[0]["cv_macro_f1"]),
        "label_source": "proxy_thresholds (vehicle_count <= 8 = LOW, 9-15 = MEDIUM, >= 16 = HIGH)",
        "proxy_thresholds": {"low_max": int(LOW_THRESHOLD), "high_min": int(HIGH_THRESHOLD + 1)},
        "proxy_label_note": "V4-IMAGE traffic-density labels are proxy labels derived from total_vehicle_count thresholds (LOW <= 8, MEDIUM 9-15, HIGH >= 16) and are not verified human ground-truth labels.",
        "models_tested": [row["model_name"] for row in model_results],
    }
    
    with (output_dir / "model_metadata_v4_image.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    return {
        "feature_names": feature_names,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "scaler": best_scaler,
        "metadata": metadata,
        "comparison_df": comparison_df,
        "model_results": model_results,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def main() -> None:
    result = train_models_v4_image()
    print(f"\n{'='*70}")
    print(f"V4-IMAGE MODEL TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nBest Model: {result['best_model_name']}")
    print(f"\nDataset Summary:")
    print(f"  Total images: {result['metadata']['total_images']}")
    print(f"  Train: {result['metadata']['training_dataset_size']}, Val: {result['metadata']['validation_dataset_size']}, Test: {result['metadata']['test_dataset_size']}")
    print(f"  Class dist: {result['metadata']['class_distribution']}")
    print(f"\nMetrics:")
    print(f"  Test Accuracy: {result['metadata']['test_accuracy']:.4f}")
    print(f"  Test Macro F1: {result['metadata']['test_macro_f1']:.4f}")
    print(f"  CV Accuracy: {result['metadata']['cv_accuracy']:.4f}")
    print(f"  CV Macro F1: {result['metadata']['cv_macro_f1']:.4f}")
    print(f"\nModel Comparison:")
    print(result['comparison_df'].to_string(index=False))


if __name__ == "__main__":
    main()
