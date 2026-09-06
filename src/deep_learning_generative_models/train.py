"""Explicit training entry point for the Convolutional Autoencoder."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from deep_learning_generative_models.config import (
    ExperimentConfig,
    load_config,
    load_default_config,
)
from deep_learning_generative_models.data import build_fashion_mnist_loaders
from deep_learning_generative_models.device import DeviceInfo, get_device
from deep_learning_generative_models.experiments import ExperimentRecord, create_experiment
from deep_learning_generative_models.models import ConvolutionalAutoencoder, build_model
from deep_learning_generative_models.reproducibility import seed_everything

HISTORY_FILENAME = "training_history.csv"
CHECKPOINT_FILENAME = "checkpoint.pt"


@dataclass(frozen=True)
class EpochHistory:
    epoch: int
    train_reconstruction_loss: float


@dataclass(frozen=True)
class TrainingResult:
    experiment: ExperimentRecord
    history: list[EpochHistory]
    checkpoint_path: Path
    history_path: Path
    final_loss: float
    duration_seconds: float
    device_info: DeviceInfo
    train_dataset_size: int


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
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


def train_autoencoder(
    config: ExperimentConfig,
    train_loader: DataLoader,
    device_info: DeviceInfo,
    experiment: ExperimentRecord,
) -> TrainingResult:
    if config.model_type != "ae":
        raise ValueError("Autoencoder training requires model_type='ae'")

    model = build_model(config).to(device_info.device)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    history: list[EpochHistory] = []

    started_at = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            device=device_info.device,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )
        history.append(
            EpochHistory(epoch=epoch, train_reconstruction_loss=epoch_loss)
        )
        print(
            f"Epoch {epoch}/{config.epochs} - "
            f"train reconstruction loss: {epoch_loss:.6f}"
        )

    duration_seconds = time.perf_counter() - started_at
    history_path = experiment.path / HISTORY_FILENAME
    checkpoint_path = experiment.path / CHECKPOINT_FILENAME
    save_training_history(history, history_path)
    save_autoencoder_checkpoint(
        model=model,
        config=config,
        history=history,
        checkpoint_path=checkpoint_path,
    )

    return TrainingResult(
        experiment=experiment,
        history=history,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        final_loss=history[-1].train_reconstruction_loss,
        duration_seconds=duration_seconds,
        device_info=device_info,
        train_dataset_size=len(train_loader.dataset),
    )


def save_training_history(history: Iterable[EpochHistory], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_reconstruction_loss"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row.epoch,
                    "train_reconstruction_loss": row.train_reconstruction_loss,
                }
            )


def save_autoencoder_checkpoint(
    model: ConvolutionalAutoencoder,
    config: ExperimentConfig,
    history: list[EpochHistory],
    checkpoint_path: Path,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": config.model_type,
            "architecture_preset": config.architecture_preset,
            "latent_dim": config.latent_dim,
            "epochs": config.epochs,
            "config": config.to_dict(),
            "history": [
                {
                    "epoch": row.epoch,
                    "train_reconstruction_loss": row.train_reconstruction_loss,
                }
                for row in history
            ],
        },
        checkpoint_path,
    )


def load_autoencoder_checkpoint(
    checkpoint_path: Path | str,
    map_location: str | torch.device = "cpu",
) -> tuple[ConvolutionalAutoencoder, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if checkpoint.get("model_type") != "ae":
        raise ValueError("Checkpoint is not an Autoencoder checkpoint")

    config = ExperimentConfig(**checkpoint["config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def run_training(config: ExperimentConfig) -> TrainingResult:
    if config.model_type != "ae":
        raise ValueError("Only AE training is implemented at this milestone")

    device_info = get_device()
    seed_everything(config.random_seed)
    experiment = create_experiment(config, device_info)
    loaders = build_fashion_mnist_loaders(config)
    return train_autoencoder(
        config=config,
        train_loader=loaders.train,
        device_info=device_info,
        experiment=experiment,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Convolutional Autoencoder. Training runs only when "
            "this command is executed."
        )
    )
    parser.add_argument("--config", help="Optional JSON config path.")
    parser.add_argument("--model", choices=["ae"], help="Model type to train.")
    parser.add_argument(
        "--preset",
        choices=["small", "medium", "deep"],
        help="Autoencoder architecture preset.",
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

    print("Autoencoder training complete.")
    print(f"Device: {result.device_info.description}")
    print(f"Train dataset size: {result.train_dataset_size}")
    print(f"Final training reconstruction loss: {result.final_loss:.6f}")
    print(f"Duration seconds: {result.duration_seconds:.2f}")
    print(f"Experiment directory: {result.experiment.path}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Training history: {result.history_path}")


if __name__ == "__main__":
    main()
