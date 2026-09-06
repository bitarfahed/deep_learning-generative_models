"""VAE generation and latent-space interpolation utilities."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from deep_learning_generative_models.config import ExperimentConfig
from deep_learning_generative_models.data import build_fashion_mnist_loaders
from deep_learning_generative_models.device import DeviceInfo, get_device
from deep_learning_generative_models.models import VariationalAutoencoder
from deep_learning_generative_models.reproducibility import seed_everything
from deep_learning_generative_models.train import CHECKPOINT_FILENAME, load_model_checkpoint

GENERATION_DIRNAME = "generation"
GENERATED_GRID_FILENAME = "generated_grid.png"
INTERPOLATION_FILENAME = "latent_interpolation.png"
SUMMARY_FILENAME = "generation_summary.json"


@dataclass(frozen=True)
class GenerationResult:
    checkpoint_path: Path
    output_dir: Path
    generated_grid_path: Path | None
    interpolation_path: Path | None
    summary_path: Path
    generated_count: int
    interpolation_steps: int
    device_info: DeviceInfo
    config: ExperimentConfig


def load_vae_checkpoint(
    checkpoint_path: Path | str,
    map_location: str | torch.device = "cpu",
) -> tuple[VariationalAutoencoder, dict[str, object]]:
    model, checkpoint = load_model_checkpoint(checkpoint_path, map_location=map_location)
    if not isinstance(model, VariationalAutoencoder):
        raise ValueError("Checkpoint is not a VAE checkpoint")
    return model, checkpoint


def sample_latent_vectors(
    count: int,
    latent_dim: int,
    device: torch.device,
    seed: int | None = None,
) -> Tensor:
    if count <= 0:
        raise ValueError("count must be positive")
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    return torch.randn(count, latent_dim, generator=generator, device=device)


def generate_images(
    model: VariationalAutoencoder,
    count: int,
    device: torch.device,
    seed: int | None = None,
) -> Tensor:
    model.eval()
    z = sample_latent_vectors(
        count=count,
        latent_dim=model.latent_dim,
        device=device,
        seed=seed,
    )
    with torch.no_grad():
        return model.decode(z).detach().cpu()


def interpolate_latents(start: Tensor, end: Tensor, steps: int) -> Tensor:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if start.shape != end.shape:
        raise ValueError("start and end latent tensors must have matching shapes")

    weights = torch.linspace(0.0, 1.0, steps, device=start.device)
    view_shape = (steps,) + (1,) * start.ndim
    return (1.0 - weights.view(view_shape)) * start.unsqueeze(0) + weights.view(
        view_shape
    ) * end.unsqueeze(0)


def interpolate_between_images(
    model: VariationalAutoencoder,
    image_a: Tensor,
    image_b: Tensor,
    steps: int,
    device: torch.device,
) -> Tensor:
    model.eval()
    image_a = _ensure_image_batch(image_a).to(device)
    image_b = _ensure_image_batch(image_b).to(device)

    with torch.no_grad():
        mu_a, _logvar_a = model.encode(image_a)
        mu_b, _logvar_b = model.encode(image_b)
        latent_path = interpolate_latents(mu_a.squeeze(0), mu_b.squeeze(0), steps)
        return model.decode(latent_path).detach().cpu()


def save_image_grid(images: Tensor, path: Path, columns: int | None = None) -> None:
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError("images must have shape [count, 1, 28, 28]")
    if images.shape[0] == 0:
        raise ValueError("At least one image is required")

    path.parent.mkdir(parents=True, exist_ok=True)
    count = images.shape[0]
    columns = columns or math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.3, rows * 1.3))
    axes = _flatten_axes(axes)

    for index, axis in enumerate(axes):
        axis.axis("off")
        if index < count:
            image = images[index].squeeze(0).detach().cpu().numpy()
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)

    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_generation(
    checkpoint_path: Path | str,
    generated_count: int = 16,
    interpolation_steps: int = 8,
    index_a: int = 0,
    index_b: int = 1,
    seed: int | None = None,
    mode: str = "both",
) -> GenerationResult:
    resolved_checkpoint_path = Path(checkpoint_path)
    if not resolved_checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint_path}")
    if mode not in {"generate", "interpolate", "both"}:
        raise ValueError("mode must be 'generate', 'interpolate', or 'both'")

    device_info = get_device()
    if seed is not None:
        seed_everything(seed)

    model, checkpoint = load_vae_checkpoint(
        resolved_checkpoint_path,
        map_location=device_info.device,
    )
    model.to(device_info.device)
    config = ExperimentConfig(**checkpoint["config"])
    output_dir = resolved_checkpoint_path.parent / GENERATION_DIRNAME
    generated_grid_path = None
    interpolation_path = None

    if mode in {"generate", "both"}:
        generated = generate_images(
            model=model,
            count=generated_count,
            device=device_info.device,
            seed=seed,
        )
        generated_grid_path = output_dir / GENERATED_GRID_FILENAME
        save_image_grid(generated, generated_grid_path)

    if mode in {"interpolate", "both"}:
        config_for_examples = replace(
            config,
            test_subset_size=max(index_a, index_b) + 1,
        )
        loaders = build_fashion_mnist_loaders(config_for_examples)
        test_dataset = getattr(loaders.test, "dataset", loaders.test)
        image_a = _dataset_image(test_dataset, index_a)
        image_b = _dataset_image(test_dataset, index_b)
        interpolation = interpolate_between_images(
            model=model,
            image_a=image_a,
            image_b=image_b,
            steps=interpolation_steps,
            device=device_info.device,
        )
        interpolation_path = output_dir / INTERPOLATION_FILENAME
        save_image_grid(interpolation, interpolation_path, columns=interpolation_steps)

    result = GenerationResult(
        checkpoint_path=resolved_checkpoint_path,
        output_dir=output_dir,
        generated_grid_path=generated_grid_path,
        interpolation_path=interpolation_path,
        summary_path=output_dir / SUMMARY_FILENAME,
        generated_count=generated_count if mode in {"generate", "both"} else 0,
        interpolation_steps=interpolation_steps if mode in {"interpolate", "both"} else 0,
        device_info=device_info,
        config=config,
    )
    save_generation_summary(result, checkpoint, index_a=index_a, index_b=index_b, mode=mode)
    return result


def save_generation_summary(
    result: GenerationResult,
    checkpoint: dict[str, object],
    index_a: int,
    index_b: int,
    mode: str,
) -> None:
    summary = {
        "checkpoint_path": str(result.checkpoint_path),
        "model_type": checkpoint["model_type"],
        "architecture_preset": checkpoint["architecture_preset"],
        "latent_dim": checkpoint["latent_dim"],
        "device": str(result.device_info.device),
        "device_description": result.device_info.description,
        "mode": mode,
        "generated_count": result.generated_count,
        "interpolation_steps": result.interpolation_steps,
        "interpolation_indices": [index_a, index_b],
        "generated_grid_path": (
            str(result.generated_grid_path)
            if result.generated_grid_path is not None
            else None
        ),
        "interpolation_path": (
            str(result.interpolation_path)
            if result.interpolation_path is not None
            else None
        ),
    }
    result.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with result.summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")


def _dataset_image(dataset: object, index: int) -> Tensor:
    if index < 0:
        raise ValueError("image indices must be non-negative")
    image, _label = dataset[index]  # type: ignore[index]
    if not isinstance(image, Tensor):
        raise TypeError("Dataset image must be a tensor")
    return image


def _ensure_image_batch(image: Tensor) -> Tensor:
    if image.shape == (1, 28, 28):
        return image.unsqueeze(0)
    if image.shape == (1, 1, 28, 28):
        return image
    raise ValueError("image must have shape [1, 28, 28] or [1, 1, 28, 28]")


def _flatten_axes(axes: object) -> list[object]:
    if hasattr(axes, "flat"):
        return list(axes.flat)  # type: ignore[attr-defined]
    return [axes]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate VAE samples and latent interpolations from a checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help=f"Path to {CHECKPOINT_FILENAME}.")
    parser.add_argument(
        "--mode",
        choices=["generate", "interpolate", "both"],
        default="both",
        help="Generation operation to run.",
    )
    parser.add_argument("--count", type=int, default=16, help="Generated image count.")
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Latent interpolation step count.",
    )
    parser.add_argument("--index-a", type=int, default=0, help="First test image index.")
    parser.add_argument("--index-b", type=int, default=1, help="Second test image index.")
    parser.add_argument("--seed", type=int, help="Optional random seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_generation(
        checkpoint_path=args.checkpoint,
        generated_count=args.count,
        interpolation_steps=args.steps,
        index_a=args.index_a,
        index_b=args.index_b,
        seed=args.seed,
        mode=args.mode,
    )

    print("VAE generation complete.")
    print(f"Device: {result.device_info.description}")
    print(f"Output directory: {result.output_dir}")
    print(f"Generated grid: {result.generated_grid_path}")
    print(f"Interpolation: {result.interpolation_path}")
    print(f"Summary: {result.summary_path}")


if __name__ == "__main__":
    main()
