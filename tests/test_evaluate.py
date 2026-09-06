from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from deep_learning_generative_models.config import ExperimentConfig
from deep_learning_generative_models.evaluate import (
    EVALUATION_DIRNAME,
    evaluate_checkpoint,
    evaluate_reconstruction_loss,
    save_reconstruction_figure,
    save_training_loss_plot,
)
from deep_learning_generative_models.models import build_model
from deep_learning_generative_models.train import (
    EpochHistory,
    LEGACY_HISTORY_FILENAME,
    save_autoencoder_checkpoint,
)


def _config() -> ExperimentConfig:
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


def _loader(batch_size: int = 2) -> DataLoader:
    images = torch.rand(4, 1, 28, 28)
    labels = torch.zeros(4, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def _checkpoint_path(tmp_path) -> tuple[ExperimentConfig, torch.nn.Module, object]:
    config = _config()
    model = build_model(config)
    history = [EpochHistory(epoch=1, train_reconstruction_loss=0.123)]
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_autoencoder_checkpoint(model, config, history, checkpoint_path)
    return config, model, checkpoint_path


def test_evaluate_reconstruction_loss_is_finite_and_uses_no_grad() -> None:
    class NoGradAwareModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.grad_enabled_during_forward: bool | None = None

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            self.grad_enabled_during_forward = torch.is_grad_enabled()
            return images

    model = NoGradAwareModel()

    loss, samples = evaluate_reconstruction_loss(
        model=model,
        data_loader=_loader(),
        device=torch.device("cpu"),
    )

    assert torch.isfinite(torch.tensor(loss))
    assert samples == 4
    assert model.grad_enabled_during_forward is False


def test_reconstruction_figure_is_created(tmp_path) -> None:
    path = tmp_path / "reconstructions.png"

    save_reconstruction_figure(
        originals=torch.rand(2, 1, 28, 28),
        reconstructions=torch.rand(2, 1, 28, 28),
        path=path,
    )

    assert path.is_file()
    assert path.stat().st_size > 0


def test_training_loss_plot_is_created(tmp_path) -> None:
    path = tmp_path / "training_loss.png"

    result = save_training_loss_plot(
        history=[{"epoch": 1, "train_reconstruction_loss": 0.2}],
        path=path,
    )

    assert result == path
    assert path.is_file()
    assert path.stat().st_size > 0


def test_empty_training_loss_history_skips_plot(tmp_path) -> None:
    path = tmp_path / "training_loss.png"

    result = save_training_loss_plot(history=[], path=path)

    assert result is None
    assert not path.exists()


def test_evaluate_checkpoint_creates_summary_and_outputs(monkeypatch, tmp_path) -> None:
    config, _model, checkpoint_path = _checkpoint_path(tmp_path)

    class FakeLoaders:
        train = _loader()
        test = _loader()

    monkeypatch.setattr(
        "deep_learning_generative_models.evaluate.build_fashion_mnist_loaders",
        lambda _config: FakeLoaders,
    )

    result = evaluate_checkpoint(checkpoint_path, max_samples=3, max_images=2)

    assert result.output_dir == tmp_path / EVALUATION_DIRNAME
    assert result.reconstruction_figure_path.is_file()
    assert result.training_loss_plot_path is not None
    assert result.training_loss_plot_path.is_file()
    assert result.summary_path.is_file()
    assert result.evaluated_samples == 4
    assert result.config.test_subset_size == 3

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["model_type"] == "ae"
    assert summary["architecture_preset"] == config.architecture_preset
    assert summary["latent_dim"] == config.latent_dim
    assert summary["evaluated_samples"] == 4
    assert summary["test_reconstruction_loss"] == result.test_reconstruction_loss
    assert summary["reconstruction_figure_path"] == str(
        result.reconstruction_figure_path
    )


def test_evaluate_checkpoint_loads_legacy_history_file(monkeypatch, tmp_path) -> None:
    _config, _model, checkpoint_path = _checkpoint_path(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["history"] = []
    torch.save(checkpoint, checkpoint_path)
    (tmp_path / LEGACY_HISTORY_FILENAME).write_text(
        "epoch,train_reconstruction_loss\n1,0.2\n",
        encoding="utf-8",
    )

    class FakeLoaders:
        train = _loader()
        test = _loader()

    monkeypatch.setattr(
        "deep_learning_generative_models.evaluate.build_fashion_mnist_loaders",
        lambda _config: FakeLoaders,
    )

    result = evaluate_checkpoint(checkpoint_path, max_samples=3, max_images=2)

    assert result.training_loss_plot_path is not None
    assert result.training_loss_plot_path.is_file()


def test_evaluate_checkpoint_rejects_missing_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        evaluate_checkpoint(tmp_path / "missing.pt")


def test_evaluate_checkpoint_rejects_invalid_sample_limit(tmp_path) -> None:
    _config, _model, checkpoint_path = _checkpoint_path(tmp_path)

    with pytest.raises(ValueError, match="max_samples must be positive"):
        evaluate_checkpoint(checkpoint_path, max_samples=0)
