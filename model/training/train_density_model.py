from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "datasets" / "final"
MODEL_DIR = ROOT / "model" / "saved_models"
EVAL_DIR = ROOT / "model" / "evaluation"
DISALLOWED_FEATURES = {
    "traffic_density",
    "avg_total_vehicle_count",
    "congestion_score",
    "video_id",
    "video_name",
    "window_start",
    "window_end",
}
TARGET_COLUMN = "traffic_density"
LABEL_ORDER = ["LOW", "MEDIUM", "HIGH"]


def get_feature_columns(columns: list[str] | pd.Index | tuple[str, ...]) -> list[str]:
    """Return the model input feature list while excluding the target and leakage columns."""
    feature_names: list[str] = []
    for column in columns:
        name = str(column)
        if name in DISALLOWED_FEATURES or name == TARGET_COLUMN:
            continue
        if name not in feature_names:
            feature_names.append(name)
    return feature_names


def load_split(split_name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}' in {path.name}")
    return df


def make_model_specs() -> list[tuple[str, object]]:
    return [
        ("Logistic Regression", LogisticRegression(max_iter=5000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced_subsample")),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=400, random_state=42, class_weight="balanced")),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
    ]


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=LABEL_ORDER, zero_division=0)),
    }


def build_feature_importance(model: object, feature_names: list[str]) -> list[dict[str, str | float]]:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(feature_names, importances, strict=True)
        ]
    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)
        if coefficients.ndim == 1:
            importances = np.abs(coefficients)
        else:
            importances = np.abs(coefficients).mean(axis=0)
        return [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(feature_names, importances, strict=True)
        ]
    return [{"feature": feature, "importance": 0.0} for feature in feature_names]


def train_models() -> dict[str, object]:
    train_df = load_split("train")
    val_df = load_split("validation")
    test_df = load_split("test")

    feature_names = get_feature_columns(train_df.columns)
    if not feature_names:
        raise ValueError("No valid feature columns were found after excluding target and leaked columns.")

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

    results: list[dict[str, object]] = []
    trained_models: dict[str, object] = {}

    for model_name, estimator in make_model_specs():
        model = estimator
        model.fit(X_train_scaled, y_train)
        val_pred = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)

        model_results = {
            "model_name": model_name,
            "validation_metrics": compute_metrics(y_val, val_pred),
            "test_metrics": compute_metrics(y_test, test_pred),
            "validation_confusion_matrix": confusion_matrix(y_val, val_pred, labels=LABEL_ORDER).tolist(),
            "test_confusion_matrix": confusion_matrix(y_test, test_pred, labels=LABEL_ORDER).tolist(),
            "validation_report": classification_report(y_val, val_pred, labels=LABEL_ORDER, output_dict=True, zero_division=0),
            "test_report": classification_report(y_test, test_pred, labels=LABEL_ORDER, output_dict=True, zero_division=0),
            "feature_importance": build_feature_importance(model, feature_names),
        }
        results.append(model_results)
        trained_models[model_name] = model

    comparison_rows: list[dict[str, object]] = []
    for row in results:
        comparison_rows.append(
            {
                "model_name": row["model_name"],
                "validation_accuracy": row["validation_metrics"]["accuracy"],
                "validation_macro_f1": row["validation_metrics"]["macro_f1"],
                "validation_weighted_f1": row["validation_metrics"]["weighted_f1"],
                "test_accuracy": row["test_metrics"]["accuracy"],
                "test_macro_f1": row["test_metrics"]["macro_f1"],
                "test_weighted_f1": row["test_metrics"]["weighted_f1"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    best_model_name = comparison_df.sort_values(["validation_macro_f1", "validation_accuracy"], ascending=False).iloc[0]["model_name"]
    best_model = trained_models[best_model_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(EVAL_DIR / "model_comparison.csv", index=False)

    importance_df = pd.DataFrame(
        [
            {"model_name": row["model_name"], "feature": item["feature"], "importance": item["importance"]}
            for row in results
            for item in row["feature_importance"]
        ]
    )
    importance_df.to_csv(EVAL_DIR / "feature_importance.csv", index=False)

    for row in results:
        model_name = str(row["model_name"])
        pd.DataFrame(row["validation_confusion_matrix"], index=LABEL_ORDER, columns=LABEL_ORDER).to_csv(
            EVAL_DIR / f"{model_name.lower().replace(' ', '_')}_validation_confusion_matrix.csv",
            index=True,
        )
        pd.DataFrame(row["test_confusion_matrix"], index=LABEL_ORDER, columns=LABEL_ORDER).to_csv(
            EVAL_DIR / f"{model_name.lower().replace(' ', '_')}_test_confusion_matrix.csv",
            index=True,
        )

    with (MODEL_DIR / "feature_names.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)

    with (MODEL_DIR / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)

    with (MODEL_DIR / "traffic_density_model.pkl").open("wb") as handle:
        pickle.dump(best_model, handle)

    metadata = {
        "target_column": TARGET_COLUMN,
        "label_order": LABEL_ORDER,
        "feature_names": feature_names,
        "best_model_name": best_model_name,
        "proxy_label_note": "The current labels are proxy labels derived from avg_total_vehicle_count and are not verified human ground-truth labels.",
        "data_files": {
            "train": str(DATA_DIR / "train.csv"),
            "validation": str(DATA_DIR / "validation.csv"),
            "test": str(DATA_DIR / "test.csv"),
        },
        "validation_metrics": {best_model_name: {"accuracy": comparison_df.loc[comparison_df["model_name"] == best_model_name, "validation_accuracy"].iloc[0], "macro_f1": comparison_df.loc[comparison_df["model_name"] == best_model_name, "validation_macro_f1"].iloc[0]}},
        "test_metrics": {best_model_name: {"accuracy": comparison_df.loc[comparison_df["model_name"] == best_model_name, "test_accuracy"].iloc[0], "macro_f1": comparison_df.loc[comparison_df["model_name"] == best_model_name, "test_macro_f1"].iloc[0]}},
    }
    with (MODEL_DIR / "model_metadata.json").open("w", encoding="utf-8") as handle:
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
    result = train_models()
    print("Selected best model:", result["best_model_name"])
    print(result["comparison_df"].to_string(index=False))


if __name__ == "__main__":
    main()
