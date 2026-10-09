from __future__ import annotations

from pathlib import Path

from model.training.train_density_model_v2 import train_models_v2
from model.training.train_density_model_v3 import train_models_v3


def test_v2_training_pipeline_saves_separate_model(tmp_path: Path) -> None:
    result = train_models_v2(output_dir=tmp_path)

    assert result["best_model_name"]
    assert (tmp_path / "traffic_density_model_v2.pkl").exists()
    assert (tmp_path / "scaler_v2.pkl").exists()
    assert (tmp_path / "feature_names_v2.json").exists()
    assert (tmp_path / "model_metadata_v2.json").exists()


def test_v3_training_pipeline_saves_separate_model(tmp_path: Path) -> None:
    result = train_models_v3(output_dir=tmp_path)

    assert result["best_model_name"]
    assert (tmp_path / "traffic_density_model_v3.pkl").exists()
    assert (tmp_path / "scaler_v3.pkl").exists()
    assert (tmp_path / "feature_names_v3.json").exists()
    assert (tmp_path / "model_metadata_v3.json").exists()
