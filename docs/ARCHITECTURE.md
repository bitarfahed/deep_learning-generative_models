# Architecture Notes

## Purpose

This document records intended component boundaries and data flow at a high level. It does not describe completed implementation.

## Decisions Already Made

- The Python package is `deep_learning_generative_models`.
- The project uses PyTorch, torchvision, matplotlib, and pytest.
- Fashion-MNIST is the planned dataset.
- The model progression is Convolutional Autoencoder first, then Variational Autoencoder.
- Heavy local artifacts such as datasets, checkpoints, and experiment runs remain outside Git.

## Planned Component Boundaries

Future implementation should separate these responsibilities:

- data access and transforms
- model definitions
- training orchestration
- evaluation and comparison
- generation and latent-space utilities
- plotting and visualization helpers
- later GUI exploration layer

The current repository intentionally contains only a package marker. These component modules should be introduced when their milestone begins.

## Planned Data Flow

At a high level, future runs should follow this flow:

1. Load explicit configuration.
2. Resolve device selection, using GPU when compatible and CPU otherwise.
3. Download or reuse cached Fashion-MNIST data.
4. Build the selected model type and architecture preset.
5. Train or load a checkpoint, depending on the command and available artifacts.
6. Save experiment configuration, checkpoint, loss history, metrics, and representative outputs.
7. Use saved models and outputs for evaluation, comparison, generation, and later GUI exploration.

## Deferred Implementation Details

The following are intentionally not decided here:

- exact neural-network layers
- channel counts and latent dimensions
- final hyperparameters
- exact configuration file format
- experiment directory schema
- GUI framework
- GUI layout and interactions

These choices belong to later implementation milestones.
