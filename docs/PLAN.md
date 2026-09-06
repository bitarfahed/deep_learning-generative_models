# Project Plan

## Objective

Build an interactive portfolio project demonstrating both Deep Learning and Generative Modeling through Fashion-MNIST image reconstruction, latent representations, and generation.

The project should prioritize educational value, clear architecture, reproducible experiments, and understandable behavior over visually impressive generated images.

## Technology Choices

Decided:

- Python
- PyTorch
- torchvision
- matplotlib
- pytest

Not currently required:

- TensorBoard
- TensorFlow
- Heavy configuration frameworks
- Additional ML frameworks

## Dataset

Decided:

- Fashion-MNIST will be the project dataset.
- Dataset files download automatically when needed.
- Dataset files are cached locally.
- Dataset files must not be committed to Git.

Deferred:

- Train/validation split policy

Fashion-MNIST loading is implemented with `ToTensor()` preprocessing and explicit optional train/test subsets.

## Model Progression

Decided:

- The baseline model will be a Convolutional Autoencoder (AE).
- The AE is a meaningful project stage for representation learning and reconstruction, not throwaway scaffolding.
- The main generative model will be a Variational Autoencoder (VAE).
- The VAE supports reconstruction, latent-space sampling, image generation, interpolation, and interactive latent-space exploration.
- GANs, diffusion models, transformers, and LLMs are outside the project scope.

Implemented architecture policy:

- AE and VAE provide three fixed architecture presets: Small, Medium, and Deep.
- Users may select a preset but should not arbitrarily modify its internal architecture.
- Each preset has a concise human-readable description.

Deferred:

- Hyperparameters

## Training and Configuration Policy

Planned behavior:

- Development training runs should normally remain short, preferably around 5-10 minutes or less on suitable hardware.
- A longer run of about 30 minutes may be acceptable when justified, but should not be required every time the application runs.
- Training results should be persisted.
- The application should load previously trained checkpoints instead of retraining whenever it starts.
- Hardware selection should detect a compatible GPU when available and fall back cleanly to CPU.

Configuration stays simple and explicit. It captures:

- model type
- architecture preset
- epochs
- batch size
- learning rate
- latent dimension where applicable
- random seed

Deferred:

- GPU verification
- Exact configuration file format
- Final deterministic behavior policy

The project should support reproducibility through explicit seeds, saved configuration, and deterministic behavior where practical, without promising perfect bit-for-bit determinism across devices and backends.

## Experiment and Output Policy

Each training experiment should eventually produce a dedicated experiment directory containing appropriate outputs, such as:

- saved configuration
- checkpoint
- loss history
- basic metrics/results
- reconstruction examples
- generated examples where applicable
- latent/interpolation outputs where applicable

Heavy or local artifacts should remain local and ignored by Git:

- downloaded datasets
- checkpoints
- experiment runs

Portfolio-safe representative figures or results may later be intentionally committed when useful for documentation.

Evaluation should remain focused on:

- reconstruction loss
- qualitative comparison of original and reconstructed/generated images
- concise AE vs VAE comparison

Additional metrics should not be added merely to make the project appear more complex.

## CLI Direction

The intended implementation direction is a clean shared package and training pipeline rather than duplicated AE/VAE scripts.

Implemented commands include:

```bash
python -m deep_learning_generative_models.train --model ae
python -m deep_learning_generative_models.train --model vae
python -m deep_learning_generative_models.evaluate --checkpoint <checkpoint>
python -m deep_learning_generative_models.generate --checkpoint <checkpoint>
python -m deep_learning_generative_models.compare --ae-checkpoint <checkpoint> --vae-checkpoint <checkpoint>
```

Training, evaluation, generation, and comparison remain explicit CLI actions.

## GUI Direction

The basic Tkinter GUI is implemented as an educational exploration layer over the completed non-GUI core.

It allows users to:

- choose between Small, Medium, and Deep fixed architecture presets
- see a short description of the selected architecture
- select Fashion-MNIST examples
- compare original images with reconstructions
- load previously trained checkpoints
- perform random generation with the VAE
- manipulate latent dimensions through controlled sliders
- interpolate between latent representations of two selected images

Training must never occur automatically merely because the GUI/application was opened.

Deferred:

- training controls, unless later explicitly justified

## Scope Boundaries

Unless a later prompt explicitly changes the scope, do not add:

- cloud deployment
- authentication
- databases
- backend/web APIs
- LLM integration
- TensorFlow
- GANs
- diffusion models
- transformers
- unrelated AI features
- unnecessary infrastructure

## Milestone Sequence

1. Repository bootstrap and Git cleanup - completed
2. Project design documentation - completed
3. Core project architecture - completed
4. Fashion-MNIST data pipeline - completed
5. Convolutional Autoencoder model - completed
6. Autoencoder training - completed
7. Autoencoder evaluation/reconstruction - completed
8. Variational Autoencoder model - completed
9. VAE training integration - completed
10. Generation and latent-space exploration core - completed
11. AE vs VAE experiments/comparison - completed
12. Engineering and QA - completed
13. Interactive GUI core - completed
14. Latent-space GUI features - completed
15. Final documentation and portfolio audit - completed

This sequence is a planning guide, not a requirement to force exactly 15 implementation prompts. Future tasks may be split or combined when technically justified, while keeping scope controlled.

## AI Assistance Disclosure

This project is being developed with significant assistance from ChatGPT and Codex for planning, documentation, code generation, review, and repository maintenance. This disclosure does not imply that AI assistance is equivalent to the author's independent implementation ability.

## Final Status

The planned portfolio implementation is complete. Feature development should stop unless a future maintenance task identifies a bug or documentation correction.
