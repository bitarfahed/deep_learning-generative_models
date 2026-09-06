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
    EpochHistory,
    LEGACY_HISTORY_FILENAME,
    compatible_training_history_paths,
    compute_loss,
    load_autoencoder_checkpoint,
    load_model_checkpoint,
    run_training,
    save_training_history,
    train_autoencoder,
    train_model,
    train_one_epoch,
    train_one_epoch_with_metrics,
    vae_loss,
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


def _vae_config() -> ExperimentConfig:
    return ExperimentConfig(
        model_type="vae",
        architecture_preset="small",
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        latent_dim=8,
        random_seed=42,
        train_subset_size=4,
        test_subset_size=4,
    )


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
    assert result.checkpoint_path.name == "ae-small-checkpoint.pt"
    assert result.history_path.name == "ae-small-training-history.csv"
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


def test_vae_loss_returns_total_reconstruction_and_kl_terms() -> None:
    config = _vae_config()
    model = build_model(config)
    images = torch.rand(2, 1, 28, 28)

    output = model(images)
    loss = vae_loss(output, images)

    assert torch.isfinite(loss.total_loss)
    assert torch.isfinite(loss.reconstruction_loss)
    assert loss.kl_loss is not None
    assert torch.isfinite(loss.kl_loss)
    assert torch.allclose(loss.total_loss, loss.reconstruction_loss + loss.kl_loss)
    assert loss.kl_loss.item() >= 0.0


def test_compute_loss_rejects_mismatched_model_output() -> None:
    with pytest.raises(TypeError, match="VAE training expected VAEForwardOutput"):
        compute_loss("vae", torch.rand(2, 1, 28, 28), torch.rand(2, 1, 28, 28))


def test_train_one_epoch_with_metrics_updates_vae_and_returns_losses() -> None:
    config = _vae_config()
    model = build_model(config)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_loss, reconstruction_loss, kl_loss = train_one_epoch_with_metrics(
        model=model,
        train_loader=_tiny_loader(),
        device=torch.device("cpu"),
        optimizer=optimizer,
        model_type="vae",
    )

    after = list(model.parameters())
    assert torch.isfinite(torch.tensor(total_loss))
    assert torch.isfinite(torch.tensor(reconstruction_loss))
    assert kl_loss is not None
    assert torch.isfinite(torch.tensor(kl_loss))
    assert any(not torch.equal(old, new) for old, new in zip(before, after))


def test_train_vae_saves_history_and_checkpoint(paths) -> None:
    config = _vae_config()
    device = DeviceInfo(torch.device("cpu"), "CPU selected for test.")
    experiment = create_experiment(config, device, paths=paths)

    result = train_model(
        config=config,
        train_loader=_tiny_loader(),
        device_info=device,
        experiment=experiment,
    )

    assert result.checkpoint_path.is_file()
    assert result.history_path.is_file()
    assert result.checkpoint_path.name == "vae-small-checkpoint.pt"
    assert result.history_path.name == "vae-small-training-history.csv"
    assert result.final_kl_loss is not None
    assert result.final_loss == result.history[-1].train_loss
    assert result.final_reconstruction_loss == result.history[-1].train_reconstruction_loss

    with result.history_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert list(rows[0]) == [
        "epoch",
        "train_loss",
        "train_reconstruction_loss",
        "train_kl_loss",
    ]
    assert rows[0]["epoch"] == "1"
    assert float(rows[0]["train_loss"]) == result.final_loss
    assert float(rows[0]["train_reconstruction_loss"]) == result.final_reconstruction_loss
    assert float(rows[0]["train_kl_loss"]) == result.final_kl_loss

    model, checkpoint = load_model_checkpoint(result.checkpoint_path)
    assert model.preset.name == "small"
    assert model.latent_dim == 8
    assert checkpoint["model_type"] == "vae"
    assert checkpoint["architecture_preset"] == "small"
    assert checkpoint["latent_dim"] == 8
    assert checkpoint["epochs"] == 1
    assert checkpoint["config"]["model_type"] == "vae"
    assert checkpoint["history"][0]["train_loss"] == result.final_loss
    assert checkpoint["history"][0]["train_kl_loss"] == result.final_kl_loss


def test_save_training_history_preserves_ae_columns(tmp_path) -> None:
    history_path = tmp_path / "training_history.csv"

    save_training_history(
        [EpochHistory(epoch=1, train_reconstruction_loss=0.25)],
        history_path,
    )

    with history_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows == [{"epoch": "1", "train_reconstruction_loss": "0.25"}]


def test_compatible_training_history_paths_include_named_then_legacy(
    config,
    tmp_path,
) -> None:
    paths = compatible_training_history_paths(tmp_path, config)

    assert [path.name for path in paths] == [
        "ae-small-training-history.csv",
        LEGACY_HISTORY_FILENAME,
    ]


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


def test_model_checkpoint_rejects_inconsistent_metadata(config, paths) -> None:
    device = DeviceInfo(torch.device("cpu"), "CPU selected for test.")
    experiment = create_experiment(config, device, paths=paths)
    result = train_autoencoder(
        config=config,
        train_loader=_tiny_loader(),
        device_info=device,
        experiment=experiment,
    )
    checkpoint = torch.load(
        result.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint["latent_dim"] = config.latent_dim + 1
    torch.save(checkpoint, result.checkpoint_path)

    with pytest.raises(ValueError, match="does not match config"):
        load_model_checkpoint(result.checkpoint_path)


def test_load_autoencoder_checkpoint_rejects_vae_checkpoint(paths) -> None:
    config = _vae_config()
    device = DeviceInfo(torch.device("cpu"), "CPU selected for test.")
    experiment = create_experiment(config, device, paths=paths)
    result = train_model(
        config=config,
        train_loader=_tiny_loader(),
        device_info=device,
        experiment=experiment,
    )

    with pytest.raises(ValueError, match="Checkpoint is not an Autoencoder checkpoint"):
        load_autoencoder_checkpoint(result.checkpoint_path)
