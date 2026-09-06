"""Smoke command for inspecting the Fashion-MNIST data pipeline."""

from __future__ import annotations

import argparse

from deep_learning_generative_models.config import load_config, load_default_config
from deep_learning_generative_models.data import (
    build_fashion_mnist_loaders,
    get_fashion_mnist_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect Fashion-MNIST DataLoaders. This does not train a model."
    )
    parser.add_argument(
        "--config",
        help="Optional path to a JSON config file. Defaults to configs/default.json.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config) if args.config else load_default_config()
    loaders = build_fashion_mnist_loaders(config)
    images, labels = next(iter(loaders.train))
    contract = get_fashion_mnist_contract()

    print("Fashion-MNIST data inspection complete.")
    print(f"Train dataset size: {len(loaders.train.dataset)}")
    print(f"Test dataset size: {len(loaders.test.dataset)}")
    print(f"Expected image shape: {contract.batch_image_shape}")
    print(f"Batch image shape: {tuple(images.shape)}")
    print(f"Label shape: {tuple(labels.shape)}")
    print(f"Image value range: {images.min().item():.4f} to {images.max().item():.4f}")


if __name__ == "__main__":
    main()
