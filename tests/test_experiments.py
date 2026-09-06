from __future__ import annotations

import json
from datetime import UTC, datetime

import torch

from deep_learning_generative_models.config import ExperimentConfig
from deep_learning_generative_models.device import DeviceInfo
from deep_learning_generative_models.experiments import (
    checkpoint_filename,
    config_filename,
    create_experiment,
    experiment_directory_name,
    metadata_filename,
    training_history_filename,
)
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
    expected_prefix = (
        "vae-medium-fashion-mnist-3ep-"
        f"{record.metadata.created_at_utc[:10].replace('-', '')}"
    )
    assert record.path.name.startswith(expected_prefix)
    assert record.path.name.endswith(record.metadata.experiment_id)
    assert record.config_path.name == "vae-medium-config.json"
    assert record.metadata_path.name == "vae-medium-metadata.json"


def test_artifact_name_helpers_are_human_readable() -> None:
    config = ExperimentConfig(
        model_type="ae",
        architecture_preset="small",
        epochs=8,
        batch_size=64,
        learning_rate=0.001,
        latent_dim=16,
        random_seed=42,
    )
    created_at = datetime(2026, 9, 6, 18, 18, 19, tzinfo=UTC)

    assert (
        experiment_directory_name(config, created_at, "9a1b0070")
        == "ae-small-fashion-mnist-8ep-20260906-181819-9a1b0070"
    )
    assert checkpoint_filename(config) == "ae-small-checkpoint.pt"
    assert training_history_filename(config) == "ae-small-training-history.csv"
    assert config_filename(config) == "ae-small-config.json"
    assert metadata_filename(config) == "ae-small-metadata.json"
