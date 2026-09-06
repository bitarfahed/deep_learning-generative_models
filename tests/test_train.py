from __future__ import annotations

import csv

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deep_learning_generative_models.config import ExperimentConfig
from deep_learning_generative_models.device import DeviceInfo
from deep_learning_generative_models.experiments import create_experiment
from deep_learning_generative_models.models import build_model
from deep_learning_generative_models.paths import ProjectPaths
from deep_learning_generative_models.train import (
    load_autoencoder_checkpoint,
    run_training,
    train_autoencoder,
    train_one_epoch,
)


@pytest.fixture
def config() -> ExperimentConfig:
    return ExperimentConfig(
        model_type="ae",
        architecture_preset="small",
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        latent_dim=8,
        random_seed=42,
        train_subset_size=4,
        test_subset_size=4,
    )


@pytest.fixture
def paths(tmp_path) -> ProjectPaths:
    return ProjectPaths(
        root=tmp_path,
        config_dir=tmp_path / "configs",
        default_config_file=tmp_path / "configs" / "default.json",
        data_dir=tmp_path / "data" / "raw",
        experiments_dir=tmp_path / "experiments",
        outputs_dir=tmp_path / "outputs",
        docs_dir=tmp_path / "docs",
        docs_results_dir=tmp_path / "docs" / "results",
    )


def _tiny_loader(batch_size: int = 2) -> DataLoader:
    images = torch.rand(4, 1, 28, 28)
    labels = torch.zeros(4, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_train_one_epoch_updates_parameters_and_returns_finite_loss(config) -> None:
    model = build_model(config)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    loss = train_one_epoch(
        model=model,
        train_loader=_tiny_loader(),
        device=torch.device("cpu"),
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
    )

    after = list(model.parameters())
    assert torch.isfinite(torch.tensor(loss))
    assert any(not torch.equal(old, new) for old, new in zip(before, after))


def test_train_autoencoder_saves_history_and_checkpoint(config, paths) -> None:
    device = DeviceInfo(torch.device("cpu"), "CPU selected for test.")
    experiment = create_experiment(config, device, paths=paths)

    result = train_autoencoder(
        config=config,
        train_loader=_tiny_loader(),
        device_info=device,
        experiment=experiment,
    )

    assert result.checkpoint_path.is_file()
    assert result.history_path.is_file()
    assert result.final_loss == result.history[-1].train_reconstruction_loss

    with result.history_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows == [
        {
            "epoch": "1",
            "train_reconstruction_loss": str(result.final_loss),
        }
    ]

    checkpoint = torch.load(
        result.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["model_type"] == "ae"
    assert checkpoint["architecture_preset"] == "small"
    assert checkpoint["latent_dim"] == 8
    assert checkpoint["epochs"] == 1
    assert "model_state_dict" in checkpoint
    assert checkpoint["config"]["train_subset_size"] == 4


def test_saved_autoencoder_checkpoint_can_be_loaded(config, paths) -> None:
    device = DeviceInfo(torch.device("cpu"), "CPU selected for test.")
    experiment = create_experiment(config, device, paths=paths)
    result = train_autoencoder(
        config=config,
        train_loader=_tiny_loader(),
        device_info=device,
        experiment=experiment,
    )

    model, checkpoint = load_autoencoder_checkpoint(result.checkpoint_path)

    assert model.preset.name == "small"
    assert model.latent_dim == 8
    assert checkpoint["model_type"] == "ae"


def test_run_training_rejects_non_ae_config(config) -> None:
    bad_config = ExperimentConfig(
        model_type="vae",
        architecture_preset=config.architecture_preset,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        latent_dim=config.latent_dim,
        random_seed=config.random_seed,
    )

    with pytest.raises(ValueError, match="Only AE training is implemented"):
        run_training(bad_config)
