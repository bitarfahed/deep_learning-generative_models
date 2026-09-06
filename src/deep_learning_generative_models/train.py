"""Explicit training entry point for Autoencoder and VAE models."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from deep_learning_generative_models.config import (
    ExperimentConfig,
    load_config,
    load_default_config,
)
from deep_learning_generative_models.data import build_fashion_mnist_loaders
from deep_learning_generative_models.device import DeviceInfo, get_device
from deep_learning_generative_models.experiments import (
    ExperimentRecord,
    checkpoint_filename,
    create_experiment,
    training_history_filename,
)
from deep_learning_generative_models.models import (
    ConvolutionalAutoencoder,
    VAEForwardOutput,
    VariationalAutoencoder,
    build_model,
)
from deep_learning_generative_models.reproducibility import seed_everything

LEGACY_HISTORY_FILENAME = "training_history.csv"
LEGACY_CHECKPOINT_FILENAME = "checkpoint.pt"
HISTORY_FILENAME = "<model>-<preset>-training-history.csv"
CHECKPOINT_FILENAME = "<model>-<preset>-checkpoint.pt"


@dataclass(frozen=True)
class EpochHistory:
    epoch: int
    train_reconstruction_loss: float
    train_loss: float | None = None
    train_kl_loss: float | None = None


@dataclass(frozen=True)
class LossBreakdown:
    total_loss: Tensor
    reconstruction_loss: Tensor
    kl_loss: Tensor | None = None


@dataclass(frozen=True)
class TrainingResult:
    experiment: ExperimentRecord
    history: list[EpochHistory]
    checkpoint_path: Path
    history_path: Path
    final_loss: float
    final_reconstruction_loss: float
    final_kl_loss: float | None
    duration_seconds: float
    device_info: DeviceInfo
    train_dataset_size: int


def autoencoder_loss(reconstructions: Tensor, targets: Tensor) -> LossBreakdown:
    reconstruction_loss = F.mse_loss(reconstructions, targets)
    return LossBreakdown(
        total_loss=reconstruction_loss,
        reconstruction_loss=reconstruction_loss,
        kl_loss=None,
    )


def vae_loss(output: VAEForwardOutput, targets: Tensor) -> LossBreakdown:
    reconstruction_loss = F.mse_loss(output.reconstruction, targets)
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + output.logvar - output.mu.pow(2) - output.logvar.exp(), dim=1)
    )
    return LossBreakdown(
        total_loss=reconstruction_loss + kl_loss,
        reconstruction_loss=reconstruction_loss,
        kl_loss=kl_loss,
    )


def compute_loss(
    model_type: str,
    model_output: Tensor | VAEForwardOutput,
    targets: Tensor,
) -> LossBreakdown:
    if model_type == "ae":
        if not isinstance(model_output, Tensor):
            raise TypeError("AE training expected tensor reconstructions")
        return autoencoder_loss(model_output, targets)
    if model_type == "vae":
        if not isinstance(model_output, VAEForwardOutput):
            raise TypeError("VAE training expected VAEForwardOutput")
        return vae_loss(model_output, targets)
    raise ValueError(f"Unsupported model type for training: {model_type!r}")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """Train one tensor-output reconstruction model for legacy AE callers."""
    model.train()
    total_loss = 0.0
    total_examples = 0

    for images, _labels in train_loader:
        images = images.to(device)

        optimizer.zero_grad()
        reconstructions = model(images)
        loss = loss_fn(reconstructions, images)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Training loader produced no examples")
    return total_loss / total_examples


def train_one_epoch_with_metrics(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    model_type: str,
) -> tuple[float, float, float | None]:
    model.train()
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_examples = 0

    for images, _labels in train_loader:
        images = images.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = compute_loss(model_type, output, images)
        loss.total_loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.total_loss.item() * batch_size
        total_reconstruction_loss += loss.reconstruction_loss.item() * batch_size
        if loss.kl_loss is not None:
            total_kl_loss += loss.kl_loss.item() * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Training loader produced no examples")

    averaged_kl_loss = total_kl_loss / total_examples if model_type == "vae" else None
    return (
        total_loss / total_examples,
        total_reconstruction_loss / total_examples,
        averaged_kl_loss,
    )


def train_model(
    config: ExperimentConfig,
    train_loader: DataLoader,
    device_info: DeviceInfo,
    experiment: ExperimentRecord,
) -> TrainingResult:
    model = build_model(config).to(device_info.device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    history: list[EpochHistory] = []

    started_at = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_loss, epoch_reconstruction_loss, epoch_kl_loss = train_one_epoch_with_metrics(
            model=model,
            train_loader=train_loader,
            device=device_info.device,
            optimizer=optimizer,
            model_type=config.model_type,
        )
        history.append(
            _build_epoch_history(
                config.model_type,
                epoch,
                epoch_loss,
                epoch_reconstruction_loss,
                epoch_kl_loss,
            )
        )
        progress = (
            f"Epoch {epoch}/{config.epochs} - "
            f"train loss: {epoch_loss:.6f}; "
            f"reconstruction loss: {epoch_reconstruction_loss:.6f}"
        )
        if epoch_kl_loss is not None:
            progress += f"; KL loss: {epoch_kl_loss:.6f}"
        print(progress)

    duration_seconds = time.perf_counter() - started_at
    history_path = experiment.path / training_history_filename(config)
    checkpoint_path = experiment.path / checkpoint_filename(config)
    save_training_history(history, history_path)
    save_model_checkpoint(
        model=model,
        config=config,
        history=history,
        checkpoint_path=checkpoint_path,
    )

    final_row = history[-1]
    return TrainingResult(
        experiment=experiment,
        history=history,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        final_loss=_history_total_loss(final_row),
        final_reconstruction_loss=final_row.train_reconstruction_loss,
        final_kl_loss=final_row.train_kl_loss,
        duration_seconds=duration_seconds,
        device_info=device_info,
        train_dataset_size=len(train_loader.dataset),
    )


def train_autoencoder(
    config: ExperimentConfig,
    train_loader: DataLoader,
    device_info: DeviceInfo,
    experiment: ExperimentRecord,
) -> TrainingResult:
    if config.model_type != "ae":
        raise ValueError("Autoencoder training requires model_type='ae'")
    return train_model(config, train_loader, device_info, experiment)


def _build_epoch_history(
    model_type: str,
    epoch: int,
    train_loss: float,
    train_reconstruction_loss: float,
    train_kl_loss: float | None,
) -> EpochHistory:
    if model_type == "ae":
        return EpochHistory(
            epoch=epoch,
            train_reconstruction_loss=train_reconstruction_loss,
        )
    if model_type == "vae":
        return EpochHistory(
            epoch=epoch,
            train_loss=train_loss,
            train_reconstruction_loss=train_reconstruction_loss,
            train_kl_loss=train_kl_loss,
        )
    raise ValueError(f"Unsupported model type for training: {model_type!r}")


def _history_total_loss(row: EpochHistory) -> float:
    return row.train_loss if row.train_loss is not None else row.train_reconstruction_loss


def _history_row(row: EpochHistory, include_vae_fields: bool) -> dict[str, float | int | None]:
    values: dict[str, float | int | None] = {
        "epoch": row.epoch,
        "train_reconstruction_loss": row.train_reconstruction_loss,
    }
    if include_vae_fields:
        values["train_loss"] = _history_total_loss(row)
        values["train_kl_loss"] = row.train_kl_loss
    return values


def save_training_history(history: Iterable[EpochHistory], path: Path) -> None:
    rows = list(history)
    include_vae_fields = any(
        row.train_loss is not None or row.train_kl_loss is not None for row in rows
    )
    fieldnames = ["epoch", "train_reconstruction_loss"]
    if include_vae_fields:
        fieldnames = ["epoch", "train_loss", "train_reconstruction_loss", "train_kl_loss"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_history_row(row, include_vae_fields))


def save_autoencoder_checkpoint(
    model: ConvolutionalAutoencoder,
    config: ExperimentConfig,
    history: list[EpochHistory],
    checkpoint_path: Path,
) -> None:
    if config.model_type != "ae":
        raise ValueError("Autoencoder checkpoint requires model_type='ae'")
    save_model_checkpoint(model, config, history, checkpoint_path)


def save_model_checkpoint(
    model: nn.Module,
    config: ExperimentConfig,
    history: list[EpochHistory],
    checkpoint_path: Path,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    include_vae_fields = config.model_type == "vae"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": config.model_type,
            "architecture_preset": config.architecture_preset,
            "latent_dim": config.latent_dim,
            "epochs": config.epochs,
            "config": config.to_dict(),
            "history": [_history_row(row, include_vae_fields) for row in history],
        },
        checkpoint_path,
    )


def load_autoencoder_checkpoint(
    checkpoint_path: Path | str,
    map_location: str | torch.device = "cpu",
) -> tuple[ConvolutionalAutoencoder, dict[str, object]]:
    model, checkpoint = load_model_checkpoint(checkpoint_path, map_location=map_location)
    if not isinstance(model, ConvolutionalAutoencoder):
        raise ValueError("Checkpoint is not an Autoencoder checkpoint")
    return model, checkpoint


def load_model_checkpoint(
    checkpoint_path: Path | str,
    map_location: str | torch.device = "cpu",
) -> tuple[ConvolutionalAutoencoder | VariationalAutoencoder, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    model_type = checkpoint.get("model_type")
    if model_type not in {"ae", "vae"}:
        raise ValueError(f"Unsupported checkpoint model type: {model_type!r}")

    config = ExperimentConfig(**checkpoint["config"])
    _validate_checkpoint_metadata(checkpoint, config)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def compatible_training_history_paths(
    experiment_dir: Path,
    config: ExperimentConfig,
) -> list[Path]:
    return [
        experiment_dir / training_history_filename(config),
        experiment_dir / LEGACY_HISTORY_FILENAME,
    ]


def _validate_checkpoint_metadata(
    checkpoint: dict[str, object],
    config: ExperimentConfig,
) -> None:
    expected = {
        "model_type": config.model_type,
        "architecture_preset": config.architecture_preset,
        "latent_dim": config.latent_dim,
        "epochs": config.epochs,
    }
    for field, expected_value in expected.items():
        if checkpoint.get(field) != expected_value:
            raise ValueError(
                f"Checkpoint metadata field {field!r} does not match config"
            )


def run_training(config: ExperimentConfig) -> TrainingResult:
    device_info = get_device()
    seed_everything(config.random_seed)
    experiment = create_experiment(config, device_info)
    loaders = build_fashion_mnist_loaders(config)
    return train_model(
        config=config,
        train_loader=loaders.train,
        device_info=device_info,
        experiment=experiment,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Autoencoder or VAE. Training runs only when "
            "this command is executed."
        )
    )
    parser.add_argument("--config", help="Optional JSON config path.")
    parser.add_argument("--model", choices=["ae", "vae"], help="Model type to train.")
    parser.add_argument(
        "--preset",
        choices=["small", "medium", "deep"],
        help="Model architecture preset.",
    )
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, help="Adam learning rate.")
    parser.add_argument("--latent-dim", type=int, help="Latent representation size.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--train-subset-size",
        type=int,
        help="Explicitly limit training examples for smoke/development runs.",
    )
    parser.add_argument(
        "--test-subset-size",
        type=int,
        help="Explicitly limit test examples in the saved config.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    config = load_config(args.config) if args.config else load_default_config()
    overrides = {
        "model_type": args.model,
        "architecture_preset": args.preset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "latent_dim": args.latent_dim,
        "random_seed": args.seed,
        "train_subset_size": args.train_subset_size,
        "test_subset_size": args.test_subset_size,
    }
    clean_overrides = {
        field: value for field, value in overrides.items() if value is not None
    }
    updated = replace(config, **clean_overrides)
    updated.validate()
    return updated


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    result = run_training(config)

    print(f"{config.model_type.upper()} training complete.")
    print(f"Device: {result.device_info.description}")
    print(f"Train dataset size: {result.train_dataset_size}")
    print(f"Final training loss: {result.final_loss:.6f}")
    print(f"Final training reconstruction loss: {result.final_reconstruction_loss:.6f}")
    if result.final_kl_loss is not None:
        print(f"Final training KL loss: {result.final_kl_loss:.6f}")
    print(f"Duration seconds: {result.duration_seconds:.2f}")
    print(f"Experiment directory: {result.experiment.path}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Training history: {result.history_path}")


if __name__ == "__main__":
    main()
