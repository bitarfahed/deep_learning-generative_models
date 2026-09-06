"""Fashion-MNIST datasets and DataLoaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from deep_learning_generative_models.config import ExperimentConfig
from deep_learning_generative_models.paths import ProjectPaths, get_project_paths

IMAGE_SHAPE: Final[tuple[int, int, int]] = (1, 28, 28)
NUM_CLASSES: Final[int] = 10


@dataclass(frozen=True)
class FashionMNISTDatasets:
    train: Dataset
    test: Dataset


@dataclass(frozen=True)
class FashionMNISTLoaders:
    train: DataLoader
    test: DataLoader


@dataclass(frozen=True)
class FashionMNISTContract:
    image_shape: tuple[int, int, int] = IMAGE_SHAPE
    batch_image_shape: tuple[str, int, int, int] = ("batch", *IMAGE_SHAPE)
    num_classes: int = NUM_CLASSES
    value_range: tuple[float, float] = (0.0, 1.0)


def build_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor()])


def build_fashion_mnist_datasets(
    config: ExperimentConfig,
    paths: ProjectPaths | None = None,
    download: bool = True,
) -> FashionMNISTDatasets:
    project_paths = paths or get_project_paths()
    transform = build_transform()

    train_dataset = datasets.FashionMNIST(
        root=str(project_paths.data_dir),
        train=True,
        transform=transform,
        download=download,
    )
    test_dataset = datasets.FashionMNIST(
        root=str(project_paths.data_dir),
        train=False,
        transform=transform,
        download=download,
    )

    return FashionMNISTDatasets(
        train=_apply_subset(train_dataset, config.train_subset_size, "train"),
        test=_apply_subset(test_dataset, config.test_subset_size, "test"),
    )


def build_fashion_mnist_loaders(
    config: ExperimentConfig,
    paths: ProjectPaths | None = None,
    download: bool = True,
) -> FashionMNISTLoaders:
    fashion_datasets = build_fashion_mnist_datasets(
        config=config,
        paths=paths,
        download=download,
    )
    return FashionMNISTLoaders(
        train=DataLoader(
            fashion_datasets.train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        ),
        test=DataLoader(
            fashion_datasets.test,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        ),
    )


def get_fashion_mnist_contract() -> FashionMNISTContract:
    return FashionMNISTContract()


def _apply_subset(dataset: Dataset, subset_size: int | None, split_name: str) -> Dataset:
    if subset_size is None:
        return dataset
    if subset_size > len(dataset):
        raise ValueError(
            f"{split_name}_subset_size={subset_size} exceeds dataset size {len(dataset)}"
        )
    return Subset(dataset, range(subset_size))
