from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from deep_learning_generative_models.compare import (
    COMPARISON_DIR_PREFIX,
    build_comparison_text,
    evaluate_model_reconstruction_loss,
    model_reconstruction,
    run_comparison,
    save_reconstruction_comparison,
    save_training_loss_comparison,
)
from deep_learning_generative_models.config import ExperimentConfig
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


def _loader(batch_size: int = 2) -> DataLoader:
    images = torch.rand(4, 1, 28, 28)
    labels = torch.zeros(4, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def _save_checkpoint(tmp_path, model_type: str):
    config = _config(model_type)
    model = build_model(config)
    checkpoint_path = tmp_path / f"{model_type}_checkpoint.pt"
    if model_type == "ae":
        history = [EpochHistory(epoch=1, train_reconstruction_loss=0.2)]
        save_autoencoder_checkpoint(model, config, history, checkpoint_path)
    else:
        history = [
            EpochHistory(
                epoch=1,
                train_loss=0.3,
                train_reconstruction_loss=0.2,
                train_kl_loss=0.1,
            )
        ]
        save_model_checkpoint(model, config, history, checkpoint_path)
    return checkpoint_path


def test_model_reconstruction_unwraps_ae_and_vae_outputs() -> None:
    images = torch.rand(2, 1, 28, 28)
    ae_reconstruction = model_reconstruction(build_model(_config("ae")), images)
    vae_reconstruction = model_reconstruction(build_model(_config("vae")), images)

    assert tuple(ae_reconstruction.shape) == (2, 1, 28, 28)
    assert tuple(vae_reconstruction.shape) == (2, 1, 28, 28)


def test_evaluate_model_reconstruction_loss_is_finite_for_vae() -> None:
    loss, samples = evaluate_model_reconstruction_loss(
        model=build_model(_config("vae")),
        data_loader=_loader(),
        device=torch.device("cpu"),
    )

    assert torch.isfinite(torch.tensor(loss))
    assert samples == 4


def test_reconstruction_comparison_figure_is_created(tmp_path) -> None:
    path = tmp_path / "comparison.png"

    save_reconstruction_comparison(
        originals=torch.rand(2, 1, 28, 28),
        ae_reconstructions=torch.rand(2, 1, 28, 28),
        vae_reconstructions=torch.rand(2, 1, 28, 28),
        path=path,
    )

    assert path.is_file()
    assert path.stat().st_size > 0


def test_training_loss_comparison_figure_is_created(tmp_path) -> None:
    path = tmp_path / "loss.png"

    result = save_training_loss_comparison(
        ae_checkpoint={"history": [{"epoch": 1, "train_reconstruction_loss": 0.2}]},
        vae_checkpoint={"history": [{"epoch": 1, "train_reconstruction_loss": 0.3}]},
        path=path,
    )

    assert result == path
    assert path.is_file()
    assert path.stat().st_size > 0


def test_build_comparison_text_reports_tradeoff_without_better_claim() -> None:
    text = build_comparison_text(ae_loss=0.1, vae_loss=0.2)

    assert "VAE reconstruction loss is higher relative to the AE" in text
    assert "VAE also regularizes its latent distribution" in text
    assert "better" not in text.lower()


def test_run_comparison_saves_summary_and_outputs(monkeypatch, tmp_path) -> None:
    ae_checkpoint = _save_checkpoint(tmp_path, "ae")
    vae_checkpoint = _save_checkpoint(tmp_path, "vae")

    class FakeLoaders:
        test = _loader()

    monkeypatch.setattr(
        "deep_learning_generative_models.compare.build_fashion_mnist_loaders",
        lambda _config: FakeLoaders,
    )

    result = run_comparison(
        ae_checkpoint_path=ae_checkpoint,
        vae_checkpoint_path=vae_checkpoint,
        max_samples=4,
        max_images=2,
        generated_count=4,
        seed=123,
        output_dir=tmp_path / "comparison-output",
    )

    assert result.output_dir == tmp_path / "comparison-output"
    assert result.summary_path.is_file()
    assert result.reconstruction_figure_path.is_file()
    assert result.training_loss_figure_path is not None
    assert result.training_loss_figure_path.is_file()
    assert result.vae_generation_path.is_file()
    assert result.evaluated_samples == 4
    assert result.generated_count == 4

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["ae"]["config"]["model_type"] == "ae"
    assert summary["vae"]["config"]["model_type"] == "vae"
    assert summary["evaluated_samples"] == 4
    assert summary["generated_count"] == 4
    assert summary["reconstruction_figure_path"] == str(
        result.reconstruction_figure_path
    )


def test_run_comparison_rejects_wrong_checkpoint_types(tmp_path) -> None:
    ae_checkpoint = _save_checkpoint(tmp_path, "ae")

    with pytest.raises(ValueError, match="Checkpoint is not a VAE checkpoint"):
        run_comparison(
            ae_checkpoint_path=ae_checkpoint,
            vae_checkpoint_path=ae_checkpoint,
            output_dir=tmp_path / "comparison-output",
        )


def test_default_comparison_directory_uses_experiments(monkeypatch, tmp_path) -> None:
    ae_checkpoint = _save_checkpoint(tmp_path, "ae")
    vae_checkpoint = _save_checkpoint(tmp_path, "vae")

    class FakeLoaders:
        test = _loader()

    class FakePaths:
        experiments_dir = tmp_path / "experiments"

    monkeypatch.setattr(
        "deep_learning_generative_models.compare.build_fashion_mnist_loaders",
        lambda _config: FakeLoaders,
    )
    monkeypatch.setattr(
        "deep_learning_generative_models.compare.get_project_paths",
        lambda: FakePaths,
    )

    result = run_comparison(
        ae_checkpoint_path=ae_checkpoint,
        vae_checkpoint_path=vae_checkpoint,
        max_samples=4,
        generated_count=2,
    )

    assert result.output_dir.parent == tmp_path / "experiments"
    assert COMPARISON_DIR_PREFIX in result.output_dir.name
