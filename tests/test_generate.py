from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import TensorDataset

from deep_learning_generative_models.config import ExperimentConfig
from deep_learning_generative_models.generate import (
    GENERATION_DIRNAME,
    generate_images,
    interpolate_between_images,
    interpolate_latents,
    load_vae_checkpoint,
    run_generation,
    sample_latent_vectors,
    save_image_grid,
)
from deep_learning_generative_models.models import VariationalAutoencoder, build_model
from deep_learning_generative_models.train import (
    EpochHistory,
    save_autoencoder_checkpoint,
    save_model_checkpoint,
)


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


def _ae_config() -> ExperimentConfig:
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


def _save_vae_checkpoint(tmp_path):
    config = _vae_config()
    model = build_model(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_model_checkpoint(
        model=model,
        config=config,
        history=[
            EpochHistory(
                epoch=1,
                train_loss=0.3,
                train_reconstruction_loss=0.2,
                train_kl_loss=0.1,
            )
        ],
        checkpoint_path=checkpoint_path,
    )
    return config, model, checkpoint_path


def test_sample_latent_vectors_is_seeded_and_shaped() -> None:
    first = sample_latent_vectors(
        count=3,
        latent_dim=5,
        device=torch.device("cpu"),
        seed=123,
    )
    second = sample_latent_vectors(
        count=3,
        latent_dim=5,
        device=torch.device("cpu"),
        seed=123,
    )

    assert tuple(first.shape) == (3, 5)
    assert torch.equal(first, second)


def test_generate_images_shape_range_and_seeded_determinism() -> None:
    model = build_model(_vae_config())

    first = generate_images(model, count=4, device=torch.device("cpu"), seed=123)
    second = generate_images(model, count=4, device=torch.device("cpu"), seed=123)

    assert tuple(first.shape) == (4, 1, 28, 28)
    assert torch.equal(first, second)
    assert torch.all(first >= 0)
    assert torch.all(first <= 1)


def test_interpolate_latents_preserves_endpoints() -> None:
    start = torch.tensor([0.0, 2.0])
    end = torch.tensor([10.0, 12.0])

    path = interpolate_latents(start, end, steps=5)

    assert tuple(path.shape) == (5, 2)
    assert torch.equal(path[0], start)
    assert torch.equal(path[-1], end)
    assert torch.equal(path[2], torch.tensor([5.0, 7.0]))


def test_interpolate_between_images_shape_and_range() -> None:
    model = build_model(_vae_config())
    image_a = torch.rand(1, 28, 28)
    image_b = torch.rand(1, 28, 28)

    images = interpolate_between_images(
        model=model,
        image_a=image_a,
        image_b=image_b,
        steps=6,
        device=torch.device("cpu"),
    )

    assert tuple(images.shape) == (6, 1, 28, 28)
    assert torch.all(images >= 0)
    assert torch.all(images <= 1)


def test_save_image_grid_creates_file(tmp_path) -> None:
    path = tmp_path / "grid.png"

    save_image_grid(torch.rand(3, 1, 28, 28), path)

    assert path.is_file()
    assert path.stat().st_size > 0


def test_load_vae_checkpoint_loads_vae_metadata(tmp_path) -> None:
    config, _model, checkpoint_path = _save_vae_checkpoint(tmp_path)

    model, checkpoint = load_vae_checkpoint(checkpoint_path)

    assert isinstance(model, VariationalAutoencoder)
    assert model.preset.name == config.architecture_preset
    assert model.latent_dim == config.latent_dim
    assert checkpoint["model_type"] == "vae"


def test_load_vae_checkpoint_rejects_ae_checkpoint(tmp_path) -> None:
    config = _ae_config()
    model = build_model(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_autoencoder_checkpoint(
        model=model,
        config=config,
        history=[EpochHistory(epoch=1, train_reconstruction_loss=0.2)],
        checkpoint_path=checkpoint_path,
    )

    with pytest.raises(ValueError, match="Checkpoint is not a VAE checkpoint"):
        load_vae_checkpoint(checkpoint_path)


def test_run_generation_saves_outputs_and_summary(monkeypatch, tmp_path) -> None:
    _config, _model, checkpoint_path = _save_vae_checkpoint(tmp_path)

    class FakeLoaders:
        test = TensorDataset(
            torch.rand(4, 1, 28, 28),
            torch.zeros(4, dtype=torch.long),
        )

    monkeypatch.setattr(
        "deep_learning_generative_models.generate.build_fashion_mnist_loaders",
        lambda _config: FakeLoaders,
    )

    result = run_generation(
        checkpoint_path=checkpoint_path,
        generated_count=4,
        interpolation_steps=5,
        index_a=0,
        index_b=1,
        seed=123,
        mode="both",
    )

    assert result.output_dir == tmp_path / GENERATION_DIRNAME
    assert result.generated_grid_path is not None
    assert result.generated_grid_path.is_file()
    assert result.interpolation_path is not None
    assert result.interpolation_path.is_file()
    assert result.summary_path.is_file()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["model_type"] == "vae"
    assert summary["generated_count"] == 4
    assert summary["interpolation_steps"] == 5
    assert summary["interpolation_indices"] == [0, 1]
