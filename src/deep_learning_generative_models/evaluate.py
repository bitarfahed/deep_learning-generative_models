"""Evaluation and reconstruction outputs for trained Autoencoder checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from deep_learning_generative_models.config import ExperimentConfig
from deep_learning_generative_models.data import build_fashion_mnist_loaders
from deep_learning_generative_models.device import DeviceInfo, get_device
from deep_learning_generative_models.train import (
    CHECKPOINT_FILENAME,
    HISTORY_FILENAME,
    load_autoencoder_checkpoint,
)

EVALUATION_DIRNAME = "evaluation"
SUMMARY_FILENAME = "evaluation_summary.json"
RECONSTRUCTION_FIGURE_FILENAME = "reconstructions.png"
TRAINING_LOSS_FIGURE_FILENAME = "training_loss.png"


@dataclass(frozen=True)
class EvaluationResult:
    checkpoint_path: Path
    output_dir: Path
    summary_path: Path
    reconstruction_figure_path: Path
    training_loss_plot_path: Path | None
    test_reconstruction_loss: float
    evaluated_samples: int
    device_info: DeviceInfo
    config: ExperimentConfig


def evaluate_reconstruction_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> tuple[float, int]:
    model.eval()
    criterion = loss_fn or nn.MSELoss()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for images, _labels in data_loader:
            images = images.to(device)
            reconstructions = model(images)
            loss = criterion(reconstructions, images)
            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Evaluation loader produced no examples")
    return total_loss / total_examples, total_examples


def save_reconstruction_figure(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    path: Path,
    max_images: int = 8,
) -> None:
    count = min(max_images, originals.shape[0], reconstructions.shape[0])
    if count <= 0:
        raise ValueError("At least one reconstruction is required for plotting")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, count, figsize=(count * 1.4, 3.0))
    if count == 1:
        axes = axes.reshape(2, 1)

    for index in range(count):
        original = originals[index].squeeze(0).detach().cpu().numpy()
        reconstruction = reconstructions[index].squeeze(0).detach().cpu().numpy()
        axes[0, index].imshow(original, cmap="gray")
        axes[0, index].axis("off")
        axes[1, index].imshow(reconstruction, cmap="gray")
        axes[1, index].axis("off")

    axes[0, 0].set_ylabel("Original", rotation=0, labelpad=28, va="center")
    axes[1, 0].set_ylabel("Recon", rotation=0, labelpad=28, va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def load_training_history_from_checkpoint(
    checkpoint: dict[str, object],
) -> list[dict[str, float | int]]:
    raw_history = checkpoint.get("history", [])
    if not isinstance(raw_history, list):
        return []

    history: list[dict[str, float | int]] = []
    for row in raw_history:
        if not isinstance(row, dict):
            continue
        history.append(
            {
                "epoch": int(row["epoch"]),
                "train_reconstruction_loss": float(row["train_reconstruction_loss"]),
            }
        )
    return history


def load_training_history_from_csv(path: Path) -> list[dict[str, float | int]]:
    if not path.is_file():
        return []

    with path.open("r", encoding="utf-8", newline="") as file:
        return [
            {
                "epoch": int(row["epoch"]),
                "train_reconstruction_loss": float(row["train_reconstruction_loss"]),
            }
            for row in csv.DictReader(file)
        ]


def save_training_loss_plot(
    history: Iterable[dict[str, float | int]],
    path: Path,
) -> Path | None:
    rows = list(history)
    if not rows:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in rows]
    losses = [float(row["train_reconstruction_loss"]) for row in rows]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(epochs, losses, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train reconstruction loss")
    ax.set_title("Autoencoder training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_evaluation_summary(result: EvaluationResult, checkpoint: dict[str, object]) -> None:
    summary = {
        "checkpoint_path": str(result.checkpoint_path),
        "model_type": checkpoint["model_type"],
        "architecture_preset": checkpoint["architecture_preset"],
        "latent_dim": checkpoint["latent_dim"],
        "device": str(result.device_info.device),
        "device_description": result.device_info.description,
        "evaluated_samples": result.evaluated_samples,
        "test_reconstruction_loss": result.test_reconstruction_loss,
        "reconstruction_figure_path": str(result.reconstruction_figure_path),
        "training_loss_plot_path": (
            str(result.training_loss_plot_path)
            if result.training_loss_plot_path is not None
            else None
        ),
    }
    result.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with result.summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")


def evaluate_checkpoint(
    checkpoint_path: Path | str,
    max_samples: int | None = None,
    max_images: int = 8,
) -> EvaluationResult:
    resolved_checkpoint_path = Path(checkpoint_path)
    if not resolved_checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint_path}")

    device_info = get_device()
    model, checkpoint = load_autoencoder_checkpoint(
        resolved_checkpoint_path,
        map_location=device_info.device,
    )
    model.to(device_info.device)
    config = ExperimentConfig(**checkpoint["config"])
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        config = replace(config, test_subset_size=max_samples)

    loaders = build_fashion_mnist_loaders(config)
    output_dir = resolved_checkpoint_path.parent / EVALUATION_DIRNAME
    summary_path = output_dir / SUMMARY_FILENAME
    reconstruction_figure_path = output_dir / RECONSTRUCTION_FIGURE_FILENAME
    training_loss_plot_path = output_dir / TRAINING_LOSS_FIGURE_FILENAME

    test_loss, evaluated_samples = evaluate_reconstruction_loss(
        model=model,
        data_loader=loaders.test,
        device=device_info.device,
    )

    with torch.no_grad():
        originals, _labels = next(iter(loaders.test))
        reconstructions = model(originals.to(device_info.device)).cpu()
    save_reconstruction_figure(
        originals=originals,
        reconstructions=reconstructions,
        path=reconstruction_figure_path,
        max_images=max_images,
    )

    history = load_training_history_from_checkpoint(checkpoint)
    if not history:
        history = load_training_history_from_csv(
            resolved_checkpoint_path.parent / HISTORY_FILENAME
        )
    saved_training_loss_plot_path = save_training_loss_plot(
        history=history,
        path=training_loss_plot_path,
    )

    result = EvaluationResult(
        checkpoint_path=resolved_checkpoint_path,
        output_dir=output_dir,
        summary_path=summary_path,
        reconstruction_figure_path=reconstruction_figure_path,
        training_loss_plot_path=saved_training_loss_plot_path,
        test_reconstruction_loss=test_loss,
        evaluated_samples=evaluated_samples,
        device_info=device_info,
        config=config,
    )
    save_evaluation_summary(result, checkpoint)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Autoencoder checkpoint without retraining."
    )
    parser.add_argument("--checkpoint", required=True, help=f"Path to {CHECKPOINT_FILENAME}.")
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Explicitly limit evaluated test samples for smoke runs.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=8,
        help="Maximum original/reconstruction pairs to plot.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        max_samples=args.max_samples,
        max_images=args.max_images,
    )

    print("Autoencoder evaluation complete.")
    print(f"Device: {result.device_info.description}")
    print(f"Evaluated samples: {result.evaluated_samples}")
    print(f"Test reconstruction loss: {result.test_reconstruction_loss:.6f}")
    print(f"Evaluation directory: {result.output_dir}")
    print(f"Reconstruction figure: {result.reconstruction_figure_path}")
    print(f"Training loss plot: {result.training_loss_plot_path}")
    print(f"Evaluation summary: {result.summary_path}")


if __name__ == "__main__":
    main()
