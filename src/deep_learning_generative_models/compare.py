"""AE vs VAE comparison workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from deep_learning_generative_models.config import ExperimentConfig
from deep_learning_generative_models.data import build_fashion_mnist_loaders
from deep_learning_generative_models.device import DeviceInfo, get_device
from deep_learning_generative_models.generate import (
    generate_images,
    load_vae_checkpoint,
    save_image_grid,
)
from deep_learning_generative_models.models import (
    VAEForwardOutput,
    VariationalAutoencoder,
)
from deep_learning_generative_models.experiments import DATASET_SLUG
from deep_learning_generative_models.paths import get_project_paths
from deep_learning_generative_models.reproducibility import seed_everything
from deep_learning_generative_models.train import (
    CHECKPOINT_FILENAME,
    load_autoencoder_checkpoint,
)

COMPARISON_DIR_PREFIX = "comparison"
SUMMARY_FILENAME = "comparison_summary.json"
RECONSTRUCTION_FIGURE_FILENAME = "reconstruction_comparison.png"
TRAINING_LOSS_FIGURE_FILENAME = "training_loss_comparison.png"
VAE_GENERATION_FILENAME = "vae_generated_grid.png"


@dataclass(frozen=True)
class ModelComparisonMetrics:
    checkpoint_path: Path
    config: ExperimentConfig
    test_reconstruction_loss: float


@dataclass(frozen=True)
class ComparisonResult:
    output_dir: Path
    summary_path: Path
    reconstruction_figure_path: Path
    training_loss_figure_path: Path | None
    vae_generation_path: Path
    ae: ModelComparisonMetrics
    vae: ModelComparisonMetrics
    evaluated_samples: int
    generated_count: int
    device_info: DeviceInfo
    comparison_text: str


def model_reconstruction(model: nn.Module, images: Tensor) -> Tensor:
    output = model(images)
    if isinstance(output, VAEForwardOutput):
        return output.reconstruction
    if isinstance(output, Tensor):
        return output
    raise TypeError("Model output must be a reconstruction tensor or VAEForwardOutput")


def evaluate_model_reconstruction_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for images, _labels in data_loader:
            images = images.to(device)
            reconstructions = model_reconstruction(model, images)
            loss = F.mse_loss(reconstructions, images)
            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Comparison loader produced no examples")
    return total_loss / total_examples, total_examples


def save_reconstruction_comparison(
    originals: Tensor,
    ae_reconstructions: Tensor,
    vae_reconstructions: Tensor,
    path: Path,
    max_images: int = 8,
) -> None:
    count = min(
        max_images,
        originals.shape[0],
        ae_reconstructions.shape[0],
        vae_reconstructions.shape[0],
    )
    if count <= 0:
        raise ValueError("At least one reconstruction is required for plotting")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, count, figsize=(count * 1.4, 4.2))
    if count == 1:
        axes = axes.reshape(3, 1)

    rows = [
        ("Original", originals),
        ("AE", ae_reconstructions),
        ("VAE", vae_reconstructions),
    ]
    for row_index, (label, images) in enumerate(rows):
        for column_index in range(count):
            axis = axes[row_index, column_index]
            image = images[column_index].squeeze(0).detach().cpu().numpy()
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")
        axes[row_index, 0].set_ylabel(label, rotation=0, labelpad=28, va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_training_loss_comparison(
    ae_checkpoint: dict[str, object],
    vae_checkpoint: dict[str, object],
    path: Path,
) -> Path | None:
    ae_history = _checkpoint_history(ae_checkpoint)
    vae_history = _checkpoint_history(vae_checkpoint)
    if not ae_history and not vae_history:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    if ae_history:
        ax.plot(
            [int(row["epoch"]) for row in ae_history],
            [float(row["train_reconstruction_loss"]) for row in ae_history],
            marker="o",
            label="AE reconstruction",
        )
    if vae_history:
        ax.plot(
            [int(row["epoch"]) for row in vae_history],
            [float(row["train_reconstruction_loss"]) for row in vae_history],
            marker="o",
            label="VAE reconstruction",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train reconstruction loss")
    ax.set_title("Training reconstruction loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def run_comparison(
    ae_checkpoint_path: Path | str,
    vae_checkpoint_path: Path | str,
    max_samples: int | None = 128,
    max_images: int = 8,
    generated_count: int = 16,
    seed: int | None = None,
    output_dir: Path | str | None = None,
) -> ComparisonResult:
    ae_path = Path(ae_checkpoint_path)
    vae_path = Path(vae_checkpoint_path)
    if not ae_path.is_file():
        raise FileNotFoundError(f"AE checkpoint not found: {ae_path}")
    if not vae_path.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if generated_count <= 0:
        raise ValueError("generated_count must be positive")

    device_info = get_device()
    if seed is not None:
        seed_everything(seed)

    ae_model, ae_checkpoint = load_autoencoder_checkpoint(
        ae_path,
        map_location=device_info.device,
    )
    vae_model, vae_checkpoint = load_vae_checkpoint(
        vae_path,
        map_location=device_info.device,
    )
    ae_model.to(device_info.device)
    vae_model.to(device_info.device)

    ae_config = ExperimentConfig(**ae_checkpoint["config"])
    vae_config = ExperimentConfig(**vae_checkpoint["config"])
    comparison_config = ae_config
    if max_samples is not None:
        comparison_config = replace(comparison_config, test_subset_size=max_samples)

    loaders = build_fashion_mnist_loaders(comparison_config)
    ae_loss, evaluated_samples = evaluate_model_reconstruction_loss(
        ae_model,
        loaders.test,
        device_info.device,
    )
    vae_loss, vae_samples = evaluate_model_reconstruction_loss(
        vae_model,
        loaders.test,
        device_info.device,
    )
    if vae_samples != evaluated_samples:
        raise ValueError("AE and VAE evaluation sample counts do not match")

    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else _default_comparison_dir(ae_config, vae_config)
    )
    reconstruction_path = resolved_output_dir / RECONSTRUCTION_FIGURE_FILENAME
    training_loss_path = resolved_output_dir / TRAINING_LOSS_FIGURE_FILENAME
    vae_generation_path = resolved_output_dir / VAE_GENERATION_FILENAME

    with torch.no_grad():
        originals, _labels = next(iter(loaders.test))
        originals_on_device = originals.to(device_info.device)
        ae_reconstructions = model_reconstruction(ae_model, originals_on_device).cpu()
        vae_reconstructions = model_reconstruction(vae_model, originals_on_device).cpu()

    save_reconstruction_comparison(
        originals=originals,
        ae_reconstructions=ae_reconstructions,
        vae_reconstructions=vae_reconstructions,
        path=reconstruction_path,
        max_images=max_images,
    )
    saved_training_loss_path = save_training_loss_comparison(
        ae_checkpoint,
        vae_checkpoint,
        training_loss_path,
    )
    generated = generate_images(
        model=vae_model,
        count=generated_count,
        device=device_info.device,
        seed=seed,
    )
    save_image_grid(generated, vae_generation_path)

    comparison_text = build_comparison_text(ae_loss, vae_loss)
    result = ComparisonResult(
        output_dir=resolved_output_dir,
        summary_path=resolved_output_dir / SUMMARY_FILENAME,
        reconstruction_figure_path=reconstruction_path,
        training_loss_figure_path=saved_training_loss_path,
        vae_generation_path=vae_generation_path,
        ae=ModelComparisonMetrics(
            checkpoint_path=ae_path,
            config=ae_config,
            test_reconstruction_loss=ae_loss,
        ),
        vae=ModelComparisonMetrics(
            checkpoint_path=vae_path,
            config=vae_config,
            test_reconstruction_loss=vae_loss,
        ),
        evaluated_samples=evaluated_samples,
        generated_count=generated_count,
        device_info=device_info,
        comparison_text=comparison_text,
    )
    save_comparison_summary(result)
    return result


def build_comparison_text(ae_loss: float, vae_loss: float) -> str:
    if ae_loss < vae_loss:
        direction = "higher"
    elif vae_loss < ae_loss:
        direction = "lower"
    else:
        direction = "equal"
    return (
        "The AE and VAE are compared on reconstruction loss and representative "
        "outputs. In this run the VAE reconstruction loss is "
        f"{direction} relative to the AE. The AE is optimized only for "
        "reconstruction, while the VAE also regularizes its latent distribution "
        "so it can decode samples from the prior."
    )


def save_comparison_summary(result: ComparisonResult) -> None:
    summary = {
        "ae": _model_metrics_to_dict(result.ae),
        "vae": _model_metrics_to_dict(result.vae),
        "device": str(result.device_info.device),
        "device_description": result.device_info.description,
        "evaluated_samples": result.evaluated_samples,
        "generated_count": result.generated_count,
        "reconstruction_figure_path": str(result.reconstruction_figure_path),
        "training_loss_figure_path": (
            str(result.training_loss_figure_path)
            if result.training_loss_figure_path is not None
            else None
        ),
        "vae_generation_path": str(result.vae_generation_path),
        "comparison": result.comparison_text,
    }
    result.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with result.summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")


def _model_metrics_to_dict(metrics: ModelComparisonMetrics) -> dict[str, object]:
    return {
        "checkpoint_path": str(metrics.checkpoint_path),
        "config": asdict(metrics.config),
        "test_reconstruction_loss": metrics.test_reconstruction_loss,
    }


def _checkpoint_history(checkpoint: dict[str, object]) -> list[dict[str, object]]:
    raw_history = checkpoint.get("history", [])
    if not isinstance(raw_history, list):
        return []
    return [row for row in raw_history if isinstance(row, dict)]


def _default_comparison_dir(
    ae_config: ExperimentConfig,
    vae_config: ExperimentConfig,
) -> Path:
    paths = get_project_paths()
    created_at = datetime.now(UTC)
    comparison_id = uuid4().hex[:8]
    directory_name = (
        f"{COMPARISON_DIR_PREFIX}-ae-{ae_config.architecture_preset}-"
        f"vs-vae-{vae_config.architecture_preset}-{DATASET_SLUG}-"
        f"{created_at:%Y%m%d-%H%M%S}-{comparison_id}"
    )
    return paths.experiments_dir / directory_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare existing AE and VAE checkpoints without retraining."
    )
    parser.add_argument(
        "--ae-checkpoint",
        required=True,
        help=f"Path to an AE {CHECKPOINT_FILENAME}.",
    )
    parser.add_argument(
        "--vae-checkpoint",
        required=True,
        help=f"Path to a VAE {CHECKPOINT_FILENAME}.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Explicitly limit evaluated test samples for smoke runs.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=8,
        help="Maximum original/reconstruction examples to plot.",
    )
    parser.add_argument(
        "--generated-count",
        type=int,
        default=16,
        help="Number of VAE prior samples to generate.",
    )
    parser.add_argument("--seed", type=int, help="Optional random seed.")
    parser.add_argument("--output-dir", help="Optional comparison output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_comparison(
        ae_checkpoint_path=args.ae_checkpoint,
        vae_checkpoint_path=args.vae_checkpoint,
        max_samples=args.max_samples,
        max_images=args.max_images,
        generated_count=args.generated_count,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print("AE vs VAE comparison complete.")
    print(f"Device: {result.device_info.description}")
    print(f"Evaluated samples: {result.evaluated_samples}")
    print(f"AE reconstruction loss: {result.ae.test_reconstruction_loss:.6f}")
    print(f"VAE reconstruction loss: {result.vae.test_reconstruction_loss:.6f}")
    print(f"Output directory: {result.output_dir}")
    print(f"Summary: {result.summary_path}")


if __name__ == "__main__":
    main()
