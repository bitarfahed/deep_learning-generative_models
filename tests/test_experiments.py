from __future__ import annotations

import json

import torch

from deep_learning_generative_models.config import ExperimentConfig
from deep_learning_generative_models.device import DeviceInfo
from deep_learning_generative_models.experiments import create_experiment
from deep_learning_generative_models.paths import ProjectPaths


def test_create_experiment_persists_config_and_metadata(tmp_path) -> None:
    paths = ProjectPaths(
        root=tmp_path,
        config_dir=tmp_path / "configs",
        default_config_file=tmp_path / "configs" / "default.json",
        data_dir=tmp_path / "data" / "raw",
        experiments_dir=tmp_path / "experiments",
        outputs_dir=tmp_path / "outputs",
        docs_dir=tmp_path / "docs",
        docs_results_dir=tmp_path / "docs" / "results",
    )
    config = ExperimentConfig(
        model_type="vae",
        architecture_preset="medium",
        epochs=3,
        batch_size=64,
        learning_rate=0.001,
        latent_dim=32,
        random_seed=123,
    )
    device = DeviceInfo(
        device=torch.device("cpu"),
        description="CPU selected for test.",
    )

    record = create_experiment(config, device, paths=paths)

    assert record.path.is_dir()
    assert record.config_path.is_file()
    assert record.metadata_path.is_file()

    saved_config = json.loads(record.config_path.read_text(encoding="utf-8"))
    saved_metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))

    assert saved_config["model_type"] == "vae"
    assert saved_config["architecture_preset"] == "medium"
    assert saved_metadata["selected_device"] == "cpu"
    assert saved_metadata["random_seed"] == 123
    expected_suffix = (
        f"{config.model_type}-{config.architecture_preset}-"
        f"{record.metadata.experiment_id}"
    )
    assert record.path.name.endswith(expected_suffix)
