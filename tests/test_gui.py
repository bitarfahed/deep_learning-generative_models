from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from deep_learning_generative_models.config import ExperimentConfig
from deep_learning_generative_models.device import DeviceInfo
from deep_learning_generative_models.gui import (
    LATENT_SLIDER_MAX,
    LATENT_SLIDER_MIN,
    ModelExplorerService,
    image_grid_tensor,
    interpolate_latent_vectors,
    modify_latent_vector,
    model_reconstruction,
    tensor_to_tk_color_rows,
)
from deep_learning_generative_models.models import build_model
from deep_learning_generative_models.train import (
    EpochHistory,
    save_autoencoder_checkpoint,
    save_model_checkpoint,
)


def _config(model_type: str) -> ExperimentConfig:
    return ExperimentConfig(
        model_type=model_type,
        architecture_preset="small",
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        latent_dim=8,
        random_seed=42,
        train_subset_size=4,
        test_subset_size=4,
    )


def _save_checkpoint(tmp_path, model_type: str):
    config = _config(model_type)
    model = build_model(config)
    checkpoint_path = tmp_path / f"{model_type}_checkpoint.pt"
    if model_type == "ae":
        save_autoencoder_checkpoint(
            model=model,
            config=config,
            history=[EpochHistory(epoch=1, train_reconstruction_loss=0.2)],
            checkpoint_path=checkpoint_path,
        )
    else:
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
    return checkpoint_path


def _fake_loader() -> DataLoader:
    images = torch.rand(6, 1, 28, 28)
    labels = torch.arange(6)
    return DataLoader(TensorDataset(images, labels), batch_size=2)


def test_model_reconstruction_unwraps_ae_and_vae_outputs() -> None:
    images = torch.rand(2, 1, 28, 28)

    ae_reconstruction = model_reconstruction(build_model(_config("ae")), images)
    vae_reconstruction = model_reconstruction(build_model(_config("vae")), images)

    assert tuple(ae_reconstruction.shape) == (2, 1, 28, 28)
    assert tuple(vae_reconstruction.shape) == (2, 1, 28, 28)


def test_tensor_to_tk_color_rows_supports_single_images_and_grids() -> None:
    rows = tensor_to_tk_color_rows(torch.zeros(1, 56, 56), scale=2)

    assert len(rows) == 112
    assert rows[0].startswith("{#000000")


def test_image_grid_tensor_combines_generated_images() -> None:
    images = torch.rand(4, 1, 28, 28)

    grid = image_grid_tensor(images, columns=2)

    assert tuple(grid.shape) == (1, 56, 56)
    assert torch.all(grid >= 0)
    assert torch.all(grid <= 1)


def test_interpolate_latent_vectors_preserves_endpoints_and_midpoint() -> None:
    start = torch.tensor([0.0, 2.0, 4.0])
    end = torch.tensor([10.0, 12.0, 14.0])

    assert torch.equal(interpolate_latent_vectors(start, end, 0.0), start)
    assert torch.equal(interpolate_latent_vectors(start, end, 1.0), end)
    assert torch.equal(
        interpolate_latent_vectors(start, end, 0.5),
        torch.tensor([5.0, 7.0, 9.0]),
    )


def test_modify_latent_vector_updates_bounded_dimensions() -> None:
    latent = torch.zeros(4)

    modified = modify_latent_vector(
        latent,
        {
            0: -10.0,
            1: 0.5,
            3: 10.0,
        },
    )

    assert modified[0] == LATENT_SLIDER_MIN
    assert modified[1] == 0.5
    assert modified[2] == 0.0
    assert modified[3] == LATENT_SLIDER_MAX
    assert torch.equal(latent, torch.zeros(4))


def test_service_loads_checkpoint_and_reconstructs(monkeypatch, tmp_path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path, "ae")

    class FakeLoaders:
        test = _fake_loader()

    monkeypatch.setattr(
        "deep_learning_generative_models.gui.build_fashion_mnist_loaders",
        lambda _config: FakeLoaders,
    )

    service = ModelExplorerService(DeviceInfo(torch.device("cpu"), "CPU test device."))
    loaded = service.load_checkpoint(
        checkpoint_path,
        expected_model_type="ae",
        expected_preset="small",
        preview_count=4,
    )
    original, reconstruction = service.reconstruct_selected(0)

    assert loaded.config.model_type == "ae"
    assert len(service.test_images) == 4
    assert tuple(original.shape) == (1, 28, 28)
    assert tuple(reconstruction.shape) == (1, 28, 28)


def test_service_rejects_incompatible_checkpoint(monkeypatch, tmp_path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path, "ae")

    class FakeLoaders:
        test = _fake_loader()

    monkeypatch.setattr(
        "deep_learning_generative_models.gui.build_fashion_mnist_loaders",
        lambda _config: FakeLoaders,
    )

    service = ModelExplorerService(DeviceInfo(torch.device("cpu"), "CPU test device."))
    with pytest.raises(ValueError, match="Checkpoint model type"):
        service.load_checkpoint(
            checkpoint_path,
            expected_model_type="vae",
            expected_preset="small",
        )


def test_service_generates_random_images_for_vae(monkeypatch, tmp_path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path, "vae")

    class FakeLoaders:
        test = _fake_loader()

    monkeypatch.setattr(
        "deep_learning_generative_models.gui.build_fashion_mnist_loaders",
        lambda _config: FakeLoaders,
    )

    service = ModelExplorerService(DeviceInfo(torch.device("cpu"), "CPU test device."))
    service.load_checkpoint(
        checkpoint_path,
        expected_model_type="vae",
        expected_preset="small",
        preview_count=4,
    )

    images = service.generate_random(count=3, seed=123)

    assert tuple(images.shape) == (3, 1, 28, 28)
    assert torch.all(images >= 0)
    assert torch.all(images <= 1)


def test_service_interpolates_selected_images_for_vae(monkeypatch, tmp_path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path, "vae")

    class FakeLoaders:
        test = _fake_loader()

    monkeypatch.setattr(
        "deep_learning_generative_models.gui.build_fashion_mnist_loaders",
        lambda _config: FakeLoaders,
    )

    service = ModelExplorerService(DeviceInfo(torch.device("cpu"), "CPU test device."))
    service.load_checkpoint(
        checkpoint_path,
        expected_model_type="vae",
        expected_preset="small",
        preview_count=4,
    )

    decoded = service.interpolate_selected_images(0, 1, 0.5)

    assert tuple(decoded.shape) == (1, 28, 28)
    assert torch.all(decoded >= 0)
    assert torch.all(decoded <= 1)


def test_service_decodes_modified_latent_for_vae(monkeypatch, tmp_path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path, "vae")

    class FakeLoaders:
        test = _fake_loader()

    monkeypatch.setattr(
        "deep_learning_generative_models.gui.build_fashion_mnist_loaders",
        lambda _config: FakeLoaders,
    )

    service = ModelExplorerService(DeviceInfo(torch.device("cpu"), "CPU test device."))
    service.load_checkpoint(
        checkpoint_path,
        expected_model_type="vae",
        expected_preset="small",
        preview_count=4,
    )
    latent = service.encode_selected_image(0)

    decoded = service.decode_modified_latent(latent, {0: 1.0, 1: -1.0})

    assert tuple(latent.shape) == (8,)
    assert tuple(decoded.shape) == (1, 28, 28)
    assert torch.all(decoded >= 0)
    assert torch.all(decoded <= 1)


def test_service_rejects_generation_for_ae(monkeypatch, tmp_path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path, "ae")

    class FakeLoaders:
        test = _fake_loader()

    monkeypatch.setattr(
        "deep_learning_generative_models.gui.build_fashion_mnist_loaders",
        lambda _config: FakeLoaders,
    )

    service = ModelExplorerService(DeviceInfo(torch.device("cpu"), "CPU test device."))
    service.load_checkpoint(
        checkpoint_path,
        expected_model_type="ae",
        expected_preset="small",
        preview_count=4,
    )

    with pytest.raises(ValueError, match="requires a VAE checkpoint"):
        service.generate_random()

    with pytest.raises(ValueError, match="requires a VAE checkpoint"):
        service.interpolate_selected_images(0, 1, 0.5)
