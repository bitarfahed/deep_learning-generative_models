# Architecture Notes

## Purpose

This document records current core infrastructure, the Fashion-MNIST data boundary, and intended future component boundaries. It does not describe model, training, evaluation, generation, or GUI implementation.

## Decisions Already Made

- The Python package is `deep_learning_generative_models`.
- The project uses PyTorch, torchvision, matplotlib, and pytest.
- Fashion-MNIST is the planned dataset.
- The model progression is Convolutional Autoencoder first, then Variational Autoencoder.
- Heavy local artifacts such as datasets, checkpoints, and experiment runs remain outside Git.

## Implemented Core Boundaries

Current infrastructure modules:

- `config.py`: JSON configuration loading, validation, and saving.
- `device.py`: PyTorch CUDA detection with CPU fallback and a human-readable device summary.
- `reproducibility.py`: seed initialization for Python random, PyTorch, and CUDA when available.
- `paths.py`: centralized `pathlib` project locations with no machine-specific absolute paths.
- `experiments.py`: unique experiment directory creation plus config and metadata persistence.
- `__main__.py`: smoke entry point for verifying the core infrastructure without training.

The default configuration is stored in `configs/default.json`.

## Implemented Data Boundary

Current data modules:

- `data.py`: Fashion-MNIST dataset construction, optional explicit subsets, DataLoader creation, and tensor contract metadata.
- `data_inspect.py`: smoke entry point for loading the real Fashion-MNIST DataLoaders and inspecting one batch without training.

The data layer uses `torchvision.datasets.FashionMNIST` with `transforms.ToTensor()` only. It downloads the dataset automatically when needed and caches it under the centralized local data path.

Tensor contract:

- image batch shape: `[batch, 1, 28, 28]`
- label batch shape: `[batch]`
- image value range after `ToTensor()`: `[0.0, 1.0]`

DataLoader policy:

- batch size comes from configuration
- training loader shuffles
- test loader does not shuffle
- `num_workers=0` for conservative Windows/PyCharm compatibility

Optional development subsets are configured explicitly through `train_subset_size` and `test_subset_size`. `null` means the full split is used.

## Planned Component Boundaries

Future implementation should separate these responsibilities:

- model definitions
- training orchestration
- evaluation and comparison
- generation and latent-space utilities
- plotting and visualization helpers
- later GUI exploration layer

Core infrastructure and Fashion-MNIST data modules exist now. Model, training, evaluation, generation, visualization, and GUI modules should be introduced when their milestone begins.

## Planned Data Flow

At a high level, future runs should follow this flow:

1. Load explicit configuration.
2. Resolve device selection, using GPU when compatible and CPU otherwise.
3. Set reproducibility seeds.
4. Create an experiment directory.
5. Save resolved configuration and metadata.
6. Load Fashion-MNIST DataLoaders when a future command needs data.
7. Later milestones will add model construction, training, checkpoints, metrics, plots, and GUI exploration.

## Deferred Implementation Details

The following are intentionally not decided here:

- exact neural-network layers
- channel counts and latent dimensions
- final hyperparameters
- complete experiment artifact schema
- GUI framework
- GUI layout and interactions

These choices belong to later implementation milestones.
