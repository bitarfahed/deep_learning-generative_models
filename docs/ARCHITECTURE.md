# Architecture Notes

## Purpose

This document records current core infrastructure, the Fashion-MNIST data boundary, Autoencoder and Variational Autoencoder model layers, AE/VAE training, AE reconstruction evaluation, VAE generation/interpolation core, AE vs VAE comparison, basic Tkinter GUI, and intended future component boundaries. It does not describe large experiment suites or advanced GUI latent controls.

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

## Implemented VAE Model Boundary

Current VAE model support lives in `models.py` beside the AE implementation.

Model interface:

- `encode(x) -> mu, logvar`
- `reparameterize(mu, logvar) -> z`
- `decode(z) -> reconstruction`
- `forward(x) -> VAEForwardOutput`

`VAEForwardOutput` contains:

- `reconstruction`
- `mu`
- `logvar`
- `z`

Latent distribution policy:

- `latent_dim` comes from the shared experiment configuration.
- `mu`, `logvar`, and sampled `z` each have shape `[batch, latent_dim]`.
- Reparameterization uses `std = exp(0.5 * logvar)`, `eps = randn_like(std)`, and `z = mu + eps * std`.
- The reparameterization path remains differentiable for future VAE training.

Output contract:

- input shape: `[batch, 1, 28, 28]`
- reconstruction shape: `[batch, 1, 28, 28]`
- reconstruction value range: `[0.0, 1.0]`
- the decoder ends with `Sigmoid`, matching Fashion-MNIST tensors produced by `ToTensor()`

VAE presets use the same convolutional channel progression as the AE presets:

- Small VAE: encoder channels `(16, 32)`, compact variational baseline.
- Medium VAE: encoder channels `(16, 32, 64)`, balanced channel capacity.
- Deep VAE: encoder channels `(32, 64, 128, 128)`, higher channel capacity.

The VAE differs from the AE by using separate linear heads for `mu` and `logvar`. It shares the same fixed preset philosophy and the same `7x7` encoded spatial feature size before latent projection.

## Implemented Training Boundary

Current training module:

- `train.py`: explicit package-style AE/VAE training command, reusable epoch loops, model-specific loss computation, checkpoint saving/loading, and CSV history persistence.

Training command:

```bash
uv run python -m deep_learning_generative_models.train --model ae
uv run python -m deep_learning_generative_models.train --model vae
```

Training occurs only when this command is run. Importing the package or opening the project does not start training.

Training policy:

- supported model types are `ae` and `vae`
- AE reconstruction loss is mean squared error
- VAE total loss is reconstruction loss plus KL-divergence loss
- VAE reconstruction loss is mean squared error
- VAE KL loss is `-0.5 * mean(sum(1 + logvar - mu^2 - exp(logvar)))`
- The current VAE loss adds mean pixel reconstruction MSE to mean per-example KL loss without a beta coefficient.
- optimizer is Adam using the configured learning rate
- images are moved to the selected device
- AE training history records average per-example reconstruction loss by epoch
- VAE training history records average total loss, reconstruction loss, and KL loss by epoch

MSE is used because Fashion-MNIST inputs are grayscale tensors in `[0.0, 1.0]`, not binary targets. AE and VAE decoders use `Sigmoid`, so outputs remain bounded for reconstruction.

Checkpoint contract:

- checkpoint file: `checkpoint.pt`
- format: `torch.save` dictionary with `state_dict`, not a serialized model object
- includes model type, architecture preset, latent dimension, epoch count, resolved config, training history, and model weights
- checkpoint loading validates top-level model metadata against the saved resolved configuration
- VAE checkpoints include enough metadata to reconstruct the selected VAE preset through the shared model factory

History contract:

- history file: `training_history.csv`
- AE columns: `epoch`, `train_reconstruction_loss`
- VAE columns: `epoch`, `train_loss`, `train_reconstruction_loss`, `train_kl_loss`

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

## Implemented VAE Generation Boundary

Current generation module:

- `generate.py`: explicit package-style VAE generation command, reusable VAE checkpoint loading, prior sampling, image decoding, latent interpolation, image-grid saving, and JSON summary persistence.

Generation command:

```bash
uv run python -m deep_learning_generative_models.generate --checkpoint experiments/<experiment-name>/checkpoint.pt --mode both
```

Generation occurs only when this command is run. Importing the package or opening the project does not generate images, load data, or retrain models.

Generation policy:

- checkpoints must be VAE checkpoints
- VAE weights are loaded from `model_state_dict`
- random samples are drawn from the standard normal latent prior
- generated samples are decoded through the trained VAE decoder
- interpolation uses the encoder mean vectors from two selected Fashion-MNIST test images
- interpolation is linear between the two latent vectors
- generated image tensors preserve shape `[count, 1, 28, 28]`
- generated image values remain in `[0.0, 1.0]`
- optional seeds make prior sampling deterministic where practical

Generation outputs are stored under the checkpoint experiment directory:

- `generation/generated_grid.png`
- `generation/latent_interpolation.png`
- `generation/generation_summary.json`

The generation summary includes checkpoint path, model type, architecture preset, latent dimension, device, mode, generated sample count, interpolation step count, selected interpolation indices, and generated artifact paths.

## Implemented AE vs VAE Comparison Boundary

Current comparison module:

- `compare.py`: explicit package-style comparison command, shared checkpoint loading, shared Fashion-MNIST test evaluation, reconstruction comparison plotting, training-loss comparison plotting, VAE generation output, and JSON summary persistence.

Comparison command:

```bash
uv run python -m deep_learning_generative_models.compare --ae-checkpoint experiments/<ae-experiment>/checkpoint.pt --vae-checkpoint experiments/<vae-experiment>/checkpoint.pt
```

Comparison occurs only when this command is run. It does not retrain either model.

Comparison policy:

- AE and VAE checkpoints are loaded from existing `checkpoint.pt` files.
- Both models are evaluated on the same Fashion-MNIST test loader policy.
- Optional `--max-samples` keeps smoke comparisons short and explicit.
- The comparison records test reconstruction loss for each model.
- Representative original/AE/VAE reconstructions are saved in one figure.
- Existing checkpoint training histories are plotted when present.
- VAE prior samples are generated to show capability beyond reconstruction.
- The summary explains the reconstruction-vs-generative tradeoff factually without claiming a universal winner.

Comparison outputs are stored under a local ignored experiment directory:

- `comparison_summary.json`
- `reconstruction_comparison.png`
- `training_loss_comparison.png`
- `vae_generated_grid.png`

## Implemented GUI Boundary

Current GUI module:

- `gui.py`: single-window Tkinter model explorer, non-visual controller service, checkpoint loading orchestration, Fashion-MNIST test-image browsing, reconstruction display, and VAE random generation display.

GUI command:

```bash
uv run python -m deep_learning_generative_models.gui
```

GUI policy:

- Tkinter is the selected GUI framework.
- The GUI loads existing checkpoints and never trains automatically.
- Checkpoint compatibility is checked against the selected model type and preset.
- The GUI uses existing device, checkpoint, data, model reconstruction, and VAE generation APIs.
- Recoverable errors are shown to the user and keep the window usable.
- Current VAE GUI behavior includes random generation only, not latent sliders or interpolation controls.

## Planned Component Boundaries

Future implementation should separate these responsibilities:

- plotting and visualization helpers
- latent-space GUI controls

Core infrastructure, Fashion-MNIST data modules, AE and VAE model layers, AE/VAE training, AE reconstruction evaluation, VAE generation/interpolation core, AE vs VAE comparison, and basic Tkinter GUI exist now. Latent-space GUI controls should be introduced when their milestone begins.

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
10. Train the VAE only through an explicit training command and save checkpoint/history artifacts.
11. Generate VAE samples and latent interpolations only from an existing VAE checkpoint.
12. Compare existing AE and VAE checkpoints without retraining.
13. Open a basic Tkinter GUI that loads existing checkpoints, reconstructs test images, and displays VAE random generation.
14. Later milestones will add latent-space GUI controls.

## Deferred Implementation Details

The following are intentionally not decided here:

- final production hyperparameters
- complete experiment artifact schema
- latent slider behavior
- interpolation GUI behavior

These choices belong to later implementation milestones.
