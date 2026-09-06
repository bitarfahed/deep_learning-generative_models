from __future__ import annotations

import pytest
import torch

from deep_learning_generative_models.config import ExperimentConfig
from deep_learning_generative_models.models import (
    AUTOENCODER_PRESETS,
    ConvolutionalAutoencoder,
    build_autoencoder,
    build_model,
    count_trainable_parameters,
    get_autoencoder_description,
)


@pytest.mark.parametrize("preset", ["small", "medium", "deep"])
def test_autoencoder_preset_construction(preset: str) -> None:
    model = build_autoencoder(preset, latent_dim=16)

    assert isinstance(model, ConvolutionalAutoencoder)
    assert model.preset.name == preset
    assert model.latent_dim == 16


@pytest.mark.parametrize("preset", ["small", "medium", "deep"])
def test_autoencoder_forward_preserves_fashion_mnist_shape(preset: str) -> None:
    model = build_autoencoder(preset, latent_dim=16)
    x = torch.rand(4, 1, 28, 28)

    reconstruction = model(x)

    assert tuple(reconstruction.shape) == tuple(x.shape)


@pytest.mark.parametrize("preset", ["small", "medium", "deep"])
def test_autoencoder_encode_and_decode_shapes(preset: str) -> None:
    model = build_autoencoder(preset, latent_dim=12)
    x = torch.rand(3, 1, 28, 28)

    z = model.encode(x)
    reconstruction = model.decode(z)

    assert tuple(z.shape) == (3, 12)
    assert tuple(reconstruction.shape) == (3, 1, 28, 28)


@pytest.mark.parametrize("preset", ["small", "medium", "deep"])
def test_autoencoder_output_range_is_bounded_by_sigmoid(preset: str) -> None:
    model = build_autoencoder(preset, latent_dim=16)
    x = torch.randn(2, 1, 28, 28)

    reconstruction = model(x)

    assert torch.all(reconstruction >= 0)
    assert torch.all(reconstruction <= 1)


def test_invalid_autoencoder_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid AE architecture preset"):
        build_autoencoder("wide", latent_dim=16)


def test_model_factory_builds_autoencoder_from_config() -> None:
    config = ExperimentConfig(
        model_type="ae",
        architecture_preset="medium",
        epochs=1,
        batch_size=8,
        learning_rate=0.001,
        latent_dim=20,
        random_seed=42,
    )

    model = build_model(config)

    assert model.preset.name == "medium"
    assert model.latent_dim == 20


def test_model_factory_rejects_unimplemented_model_type() -> None:
    config = ExperimentConfig(
        model_type="vae",
        architecture_preset="small",
        epochs=1,
        batch_size=8,
        learning_rate=0.001,
        latent_dim=16,
        random_seed=42,
    )

    with pytest.raises(ValueError, match="Only the AE model is implemented"):
        build_model(config)


def test_autoencoder_descriptions_match_presets() -> None:
    for preset_name, preset in AUTOENCODER_PRESETS.items():
        description = get_autoencoder_description(preset_name)

        assert description == preset.description
        assert preset_name.capitalize() in description
        assert str(preset.conv_block_count) in description


def test_parameter_count_increases_with_preset_capacity() -> None:
    small = count_trainable_parameters(build_autoencoder("small", latent_dim=16))
    medium = count_trainable_parameters(build_autoencoder("medium", latent_dim=16))
    deep = count_trainable_parameters(build_autoencoder("deep", latent_dim=16))

    assert small < medium < deep
