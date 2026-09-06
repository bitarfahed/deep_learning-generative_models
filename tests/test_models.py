from __future__ import annotations

import pytest
import torch

from deep_learning_generative_models.config import ExperimentConfig
from deep_learning_generative_models.models import (
    AUTOENCODER_PRESETS,
    VAE_PRESETS,
    ConvolutionalAutoencoder,
    VariationalAutoencoder,
    build_autoencoder,
    build_model,
    build_variational_autoencoder,
    count_trainable_parameters,
    get_autoencoder_description,
    get_model_description,
    get_vae_description,
    reparameterize,
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


def test_model_factory_builds_variational_autoencoder_from_config() -> None:
    config = ExperimentConfig(
        model_type="vae",
        architecture_preset="medium",
        epochs=1,
        batch_size=8,
        learning_rate=0.001,
        latent_dim=20,
        random_seed=42,
    )

    model = build_model(config)

    assert isinstance(model, VariationalAutoencoder)
    assert model.preset.name == "medium"
    assert model.latent_dim == 20


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


@pytest.mark.parametrize("preset", ["small", "medium", "deep"])
def test_vae_preset_construction(preset: str) -> None:
    model = build_variational_autoencoder(preset, latent_dim=16)

    assert isinstance(model, VariationalAutoencoder)
    assert model.preset.name == preset
    assert model.latent_dim == 16


@pytest.mark.parametrize("preset", ["small", "medium", "deep"])
def test_vae_forward_shapes(preset: str) -> None:
    model = build_variational_autoencoder(preset, latent_dim=12)
    x = torch.rand(4, 1, 28, 28)

    output = model(x)

    assert tuple(output.reconstruction.shape) == (4, 1, 28, 28)
    assert tuple(output.mu.shape) == (4, 12)
    assert tuple(output.logvar.shape) == (4, 12)
    assert tuple(output.z.shape) == (4, 12)


@pytest.mark.parametrize("preset", ["small", "medium", "deep"])
def test_vae_encode_and_decode_shapes(preset: str) -> None:
    model = build_variational_autoencoder(preset, latent_dim=10)
    x = torch.rand(3, 1, 28, 28)

    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)
    reconstruction = model.decode(z)

    assert tuple(mu.shape) == (3, 10)
    assert tuple(logvar.shape) == (3, 10)
    assert tuple(z.shape) == (3, 10)
    assert tuple(reconstruction.shape) == (3, 1, 28, 28)


@pytest.mark.parametrize("preset", ["small", "medium", "deep"])
def test_vae_output_range_is_bounded_by_sigmoid(preset: str) -> None:
    model = build_variational_autoencoder(preset, latent_dim=16)
    x = torch.randn(2, 1, 28, 28)

    output = model(x)

    assert torch.all(output.reconstruction >= 0)
    assert torch.all(output.reconstruction <= 1)


def test_reparameterization_shape_and_gradient_flow() -> None:
    mu = torch.zeros(2, 5, requires_grad=True)
    logvar = torch.zeros(2, 5, requires_grad=True)

    z = reparameterize(mu, logvar)
    z.sum().backward()

    assert tuple(z.shape) == (2, 5)
    assert mu.grad is not None
    assert logvar.grad is not None


def test_invalid_vae_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid VAE architecture preset"):
        build_variational_autoencoder("wide", latent_dim=16)


def test_vae_descriptions_match_presets() -> None:
    for preset_name, preset in VAE_PRESETS.items():
        description = get_vae_description(preset_name)

        assert description == preset.description
        assert preset_name.capitalize() in description
        assert str(preset.conv_block_count) in description


def test_shared_description_dispatches_by_model_type() -> None:
    assert get_model_description("ae", "small") == get_autoencoder_description("small")
    assert get_model_description("vae", "small") == get_vae_description("small")


def test_invalid_model_description_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid model type"):
        get_model_description("gan", "small")


def test_vae_parameter_count_increases_with_preset_capacity() -> None:
    small = count_trainable_parameters(
        build_variational_autoencoder("small", latent_dim=16)
    )
    medium = count_trainable_parameters(
        build_variational_autoencoder("medium", latent_dim=16)
    )
    deep = count_trainable_parameters(
        build_variational_autoencoder("deep", latent_dim=16)
    )

    assert small < medium < deep
