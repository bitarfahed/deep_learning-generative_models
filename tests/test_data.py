from __future__ import annotations

import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler, Subset

from deep_learning_generative_models.config import ExperimentConfig
from deep_learning_generative_models.data import (
    IMAGE_SHAPE,
    build_fashion_mnist_datasets,
    build_fashion_mnist_loaders,
    get_fashion_mnist_contract,
)
from deep_learning_generative_models.paths import ProjectPaths


class FakeFashionMNIST:
    def __init__(self, root, train, transform, download) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download
        self.size = 20 if train else 8

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        return torch.full(IMAGE_SHAPE, float(index % 2)), index % 10


@pytest.fixture
def config() -> ExperimentConfig:
    return ExperimentConfig(
        model_type="ae",
        architecture_preset="small",
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        latent_dim=16,
        random_seed=42,
        train_subset_size=6,
        test_subset_size=5,
    )


@pytest.fixture
def paths(tmp_path) -> ProjectPaths:
    return ProjectPaths(
        root=tmp_path,
        config_dir=tmp_path / "configs",
        default_config_file=tmp_path / "configs" / "default.json",
        data_dir=tmp_path / "data" / "raw",
        experiments_dir=tmp_path / "experiments",
        outputs_dir=tmp_path / "outputs",
        docs_dir=tmp_path / "docs",
        docs_results_dir=tmp_path / "docs" / "results",
    )


def test_fashion_mnist_contract() -> None:
    contract = get_fashion_mnist_contract()

    assert contract.image_shape == (1, 28, 28)
    assert contract.batch_image_shape == ("batch", 1, 28, 28)
    assert contract.num_classes == 10
    assert contract.value_range == (0.0, 1.0)


def test_build_datasets_uses_project_paths_and_subsets(monkeypatch, config, paths) -> None:
    monkeypatch.setattr(
        "deep_learning_generative_models.data.datasets.FashionMNIST",
        FakeFashionMNIST,
    )

    datasets = build_fashion_mnist_datasets(config, paths=paths, download=True)

    assert isinstance(datasets.train, Subset)
    assert isinstance(datasets.test, Subset)
    assert len(datasets.train) == 6
    assert len(datasets.test) == 5
    assert datasets.train.dataset.root == str(paths.data_dir)
    assert datasets.test.dataset.root == str(paths.data_dir)


def test_build_loaders_preserves_image_shape_and_batch_size(monkeypatch, config, paths) -> None:
    monkeypatch.setattr(
        "deep_learning_generative_models.data.datasets.FashionMNIST",
        FakeFashionMNIST,
    )

    loaders = build_fashion_mnist_loaders(config, paths=paths, download=False)
    images, labels = next(iter(loaders.train))

    assert tuple(images.shape) == (4, 1, 28, 28)
    assert tuple(labels.shape) == (4,)
    assert images.dtype == torch.float32


def test_train_loader_shuffles_and_test_loader_does_not(monkeypatch, config, paths) -> None:
    monkeypatch.setattr(
        "deep_learning_generative_models.data.datasets.FashionMNIST",
        FakeFashionMNIST,
    )

    loaders = build_fashion_mnist_loaders(config, paths=paths, download=False)

    assert isinstance(loaders.train.sampler, RandomSampler)
    assert isinstance(loaders.test.sampler, SequentialSampler)


def test_subset_size_cannot_exceed_dataset_size(monkeypatch, config, paths) -> None:
    monkeypatch.setattr(
        "deep_learning_generative_models.data.datasets.FashionMNIST",
        FakeFashionMNIST,
    )
    too_large = ExperimentConfig(
        model_type=config.model_type,
        architecture_preset=config.architecture_preset,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        latent_dim=config.latent_dim,
        random_seed=config.random_seed,
        train_subset_size=21,
        test_subset_size=None,
    )

    with pytest.raises(ValueError, match="exceeds dataset size"):
        build_fashion_mnist_datasets(too_large, paths=paths, download=False)
