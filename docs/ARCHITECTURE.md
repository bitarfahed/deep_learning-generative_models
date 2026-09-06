# Architecture Notes

## Purpose

This document records current core infrastructure, the Fashion-MNIST data boundary, the Convolutional Autoencoder model layer, Autoencoder training, AE reconstruction evaluation, and intended future component boundaries. It does not describe VAE, generation, or GUI implementation.

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

## Implemented AE Model Boundary

Current model module:

- `models.py`: Convolutional Autoencoder presets, model construction, encode/decode/forward interface, human-readable preset descriptions, and trainable parameter counting.

The AE accepts Fashion-MNIST image tensors shaped `[batch, 1, 28, 28]`.

Model interface:

- `encode(x) -> z`
- `decode(z) -> reconstruction`
- `forward(x) -> reconstruction`

Latent representation policy:

- `latent_dim` comes from the shared experiment configuration.
- The AE latent output shape is `[batch, latent_dim]`.
- No VAE-specific `mu`, `logvar`, sampling, or KL-divergence behavior exists in the AE.

Output contract:

- Reconstruction shape: `[batch, 1, 28, 28]`
- Reconstruction value range: `[0.0, 1.0]`
- The decoder ends with `Sigmoid`, matching Fashion-MNIST tensors produced by `ToTensor()`.

AE presets:

- Small: encoder channels `(16, 32)`, two stride-2 convolutional blocks, compact baseline.
- Medium: encoder channels `(16, 32, 64)`, two stride-2 blocks plus one stride-1 capacity block.
- Deep: encoder channels `(32, 64, 128, 128)`, two stride-2 blocks plus two stride-1 capacity blocks.

All presets encode to a `7x7` spatial feature map before the linear latent projection. Decoders use corresponding convolutional refinement followed by two transposed-convolution upsampling stages back to `28x28`.

## Implemented AE Training Boundary

Current training module:

- `train.py`: explicit package-style Autoencoder training command, reusable one-epoch loop, full AE training orchestration, checkpoint saving/loading, and CSV history persistence.

Training command:

```bash
uv run python -m deep_learning_generative_models.train --model ae
```

Training occurs only when this command is run. Importing the package or opening the project does not start training.

Training policy:

- model type must be `ae`
- reconstruction loss is mean squared error
- optimizer is Adam using the configured learning rate
- images are moved to the selected device
- training history records average per-example reconstruction loss by epoch

MSE is used because Fashion-MNIST inputs are grayscale tensors in `[0.0, 1.0]`, not binary targets. The AE decoder uses `Sigmoid`, so outputs remain bounded for reconstruction.

Checkpoint contract:

- checkpoint file: `checkpoint.pt`
- format: `torch.save` dictionary with `state_dict`, not a serialized model object
- includes model type, architecture preset, latent dimension, epoch count, resolved config, training history, and model weights

History contract:

- history file: `training_history.csv`
- columns: `epoch`, `train_reconstruction_loss`

Training artifacts are stored in the experiment directory and remain ignored by Git under `experiments/`.

## Implemented AE Evaluation Boundary

Current evaluation module:

- `evaluate.py`: explicit package-style Autoencoder checkpoint evaluation, test reconstruction loss calculation, reconstruction figure creation, training-loss plot creation, and JSON summary persistence.

Evaluation command:

```bash
uv run python -m deep_learning_generative_models.evaluate --checkpoint experiments/<experiment-name>/checkpoint.pt
```

Evaluation loads an existing checkpoint and does not retrain the model.

Evaluation policy:

- checkpoint must be an AE checkpoint
- model weights are loaded from `model_state_dict`
- evaluation runs under `torch.no_grad()`
- reconstruction loss is mean squared error, matching AE training
- optional `--max-samples` explicitly limits test samples for smoke runs

Evaluation outputs are stored under the checkpoint experiment directory:

- `evaluation/reconstructions.png`
- `evaluation/training_loss.png`
- `evaluation/evaluation_summary.json`

The evaluation summary includes checkpoint path, model type, architecture preset, latent dimension, device, evaluated sample count, test reconstruction loss, and generated artifact paths.

## Planned Component Boundaries

Future implementation should separate these responsibilities:

- AE vs VAE comparison
- generation and latent-space utilities
- plotting and visualization helpers
- later GUI exploration layer

Core infrastructure, Fashion-MNIST data modules, the AE model layer, AE training, and AE reconstruction evaluation exist now. VAE, generation, visualization, GUI, and comparison modules should be introduced when their milestone begins.

## Planned Data Flow

At a high level, future runs should follow this flow:

1. Load explicit configuration.
2. Resolve device selection, using GPU when compatible and CPU otherwise.
3. Set reproducibility seeds.
4. Create an experiment directory.
5. Save resolved configuration and metadata.
6. Load Fashion-MNIST DataLoaders when a future command needs data.
7. Build a Convolutional Autoencoder from the selected architecture preset when a future command needs the AE.
8. Train the AE only through an explicit training command and save checkpoint/history artifacts.
9. Evaluate trained AE checkpoints without retraining and save reconstruction artifacts.
10. Later milestones will add VAE behavior, generation, AE vs VAE comparison, and GUI exploration.

## Deferred Implementation Details

The following are intentionally not decided here:

- VAE neural-network layers
- VAE channel counts and latent policy
- final production hyperparameters
- complete experiment artifact schema
- GUI framework
- GUI layout and interactions

These choices belong to later implementation milestones.
