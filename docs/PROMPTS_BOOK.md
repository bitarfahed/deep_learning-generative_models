# Prompts Book

This append-only log records development prompts used for this repository. Repository bootstrap occurred before the Prompt Book was introduced; the exact Prompt 1 text is not recorded here because it is not available in the repository.

## Prompt 2 - Finalize Project Design Documentation

```text
PROMPT 2 — Finalize Project Design Documentation

Inspect the current repository before making any changes.

This task is documentation and project-design consolidation only.
Do NOT implement models, dataset loading, training, evaluation, generation, GUI, or other application code.

The repository bootstrap has already been completed. Do not redo the bootstrap unless you discover a small documentation/configuration inconsistency directly relevant to this task.

Goal:
Finalize and document the agreed design for the Deep Learning + Generative Models portfolio project so that future implementation prompts can follow a stable scope.

The project is intended to demonstrate both Deep Learning and Generative Modeling. The emphasis is educational value, clear architecture, reproducible experiments, and understandable behavior rather than visually impressive generated images.

Agreed project design:
- Technology: Python, PyTorch, torchvision, matplotlib, and pytest. Do not introduce TensorFlow or unnecessary frameworks. TensorBoard is not currently required.
- Dataset: Fashion-MNIST. It should eventually download automatically when needed, cache locally, and remain outside Git. No dataset implementation or download should occur in this prompt.
- Baseline: Convolutional Autoencoder (AE) for image representations and Fashion-MNIST reconstruction. The AE is a meaningful project stage and comparison baseline.
- Generative model: Variational Autoencoder (VAE), eventually supporting reconstruction, latent-space sampling, generation, interpolation, and interactive latent-space exploration.
- Architecture presets: Small, Medium, and Deep fixed presets. Users may select a preset but may not arbitrarily modify internal architecture. Do not define final layer/channel values yet unless already agreed in the repository.
- Training policy: development runs should usually remain around 5-10 minutes or less on suitable hardware; a justified longer run around 30 minutes may be acceptable. Persist results and load checkpoints rather than retraining whenever the application starts.
- Hardware: eventually detect and use a compatible GPU when possible and fall back cleanly to CPU. Do not assume current GPU support.
- Configuration: use a simple explicit configuration system capturing model type, preset, epochs, batch size, learning rate, latent dimension where applicable, and random seed. Avoid heavy configuration frameworks.
- Experiment outputs: each training experiment should eventually create its own directory with saved config, checkpoint, loss history, metrics/results, reconstruction examples, generated examples where applicable, and latent/interpolation outputs where applicable. Heavy artifacts stay local and ignored by Git; representative portfolio-safe figures/results may later be intentionally committed.
- Evaluation: focus on reconstruction loss and qualitative original/reconstructed/generated image comparison. Do not add metrics merely for complexity. AE vs VAE should receive concise meaningful comparison.
- Reproducibility: support explicit seeds, saved configuration, and deterministic behavior where practical, without promising perfect bit-for-bit determinism across all devices/backends.
- CLI direction: prefer a clean shared package and training pipeline rather than duplicated AE/VAE scripts. Future commands may resemble `python -m <package>.train --model ae` and `python -m <package>.train --model vae`. Do not implement the CLI now.
- GUI later: the GUI is an educational exploration layer over already working models, not the primary purpose. It should eventually allow preset selection and descriptions, Fashion-MNIST example selection, reconstruction comparison, checkpoint loading, VAE random generation, latent sliders, and interpolation. Training may later be an explicit user action, but must never happen automatically when the GUI opens. Do not select a GUI framework now.

Out of scope unless a later prompt explicitly changes it:
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

Milestone plan:
1. Repository bootstrap and Git cleanup — already completed
2. Project design documentation — current task
3. Core project architecture
4. Fashion-MNIST data pipeline
5. Convolutional Autoencoder model
6. Autoencoder training
7. Autoencoder evaluation/reconstruction
8. Variational Autoencoder model
9. VAE training integration
10. Generation and latent-space exploration core
11. AE vs VAE experiments/comparison
12. Engineering and QA
13. Interactive GUI core
14. Latent-space GUI features
15. Final documentation and portfolio audit

This sequence is a planning guide, not a requirement to force exactly 15 implementation prompts. Future tasks may be split or combined when technically justified, while keeping scope controlled.

Documentation work:
- Inspect README.md and docs.
- Update README.md with a concise high-level description of the agreed project direction.
- Update docs/PLAN.md with objective, technology choices, dataset, model progression, experiment/output policy, GUI direction, scope boundaries, and milestone sequence.
- Create docs/ARCHITECTURE.md only if useful, describing intended component boundaries and data flow at a high level.
- Clearly distinguish decisions already made, planned behavior, and implementation details intentionally deferred.

AI transparency:
Document that the project is being developed with significant assistance from ChatGPT and Codex. Do not imply that AI assistance is equivalent to the author's independent implementation ability.

Prompt Book:
Maintain docs/PROMPTS_BOOK.md. Append this development prompt, preserve previous entries, keep chronological order, identify this entry as Prompt 2, preserve the prompt faithfully enough for audit, and note that bootstrap occurred before Prompt Book introduction if the file does not exist. Do not reconstruct Prompt 1 unless its exact text is already available.

Verification:
Before finishing, inspect documentation for contradictions, verify agreed scope is represented accurately, verify planned behavior is not falsely described as already implemented, verify no model/training/dataset/GUI implementation was added, verify no unnecessary dependency was introduced, verify this prompt was appended to docs/PROMPTS_BOOK.md, and inspect Git status.

You may commit the completed documentation work with an appropriate commit message.

Stop after this documentation/design task. Do not begin the Core Project Architecture implementation.
```

## Prompt 3 - Core Project Architecture

```text
PROMPT 3 — Core Project Architecture

Inspect the current repository before making any changes.

This task is to implement the minimal shared project infrastructure needed before adding the Fashion-MNIST data pipeline or any neural-network model.

Do NOT implement:
- dataset downloading/loading
- Autoencoder
- VAE
- training loops
- evaluation
- generation
- GUI

Follow the existing project design in README.md, docs/PLAN.md, and docs/ARCHITECTURE.md. Do not change the agreed project scope unless a concrete implementation issue requires a small correction.

Goal:
Create a clean, minimal core architecture that future prompts can reuse for configuration, device selection, reproducibility, project paths, experiment directories, and persisted experiment metadata. Avoid over-engineering.

Inspect the existing package structure and pyproject.toml. Use the repository's existing package naming and structure where possible. Do not create duplicate package roots or unnecessary abstractions.

Configuration system:
Implement a small explicit configuration system representing at minimum model type ae/vae, architecture preset small/medium/deep, epochs, batch size, learning rate, latent dimension where applicable, and random seed. Use a simple built-in format such as JSON unless the repository justifies another lightweight format. Avoid configuration framework dependencies. Provide a default configuration for later development runs. Validation should reject invalid model type, invalid architecture preset, non-positive epochs, batch size, learning rate, and latent dimension. Do not define internal neural-network architecture of presets.

Device selection:
Implement a small utility that detects CUDA availability through the installed PyTorch environment, uses CUDA when available, otherwise falls back cleanly to CPU, exposes the selected torch.device, and provides a concise human-readable device description. Do not reinstall PyTorch or modify GPU drivers/toolchain.

Reproducibility:
Implement seed initialization for Python random, PyTorch, and CUDA seeds when CUDA is available. Use deterministic settings only where practical and do not falsely claim perfect reproducibility across hardware/backends.

Project paths:
Create centralized minimal pathlib utilities for project root, local data directory, local artifacts directory, experiment directory root, and documentation/results locations where appropriate. Paths must avoid hard-coded machine-specific absolute paths. Downloaded data, checkpoints, and experiment artifacts should remain compatible with .gitignore.

Experiment management:
Implement lightweight experiment-directory creation. Each experiment receives its own directory using readable unique naming based on timestamp, model type, architecture preset, or similar. Avoid collisions. Save resolved experiment configuration and basic metadata including experiment name/id, selected device, seed, model type, architecture preset, and creation timestamp. Do not implement metrics logging, checkpoints, plots, or model outputs.

Minimal CLI / smoke entry point:
Add only a minimal way to verify infrastructure from the terminal. It may load the default config, resolve the device, set the seed, create an experiment directory, save config/metadata, and print a concise summary. Do not implement training. Prefer package-style execution and keep the CLI small.

Tests:
Add focused pytest coverage for deterministic infrastructure behavior, including valid configuration loading, invalid configuration rejection, path resolution, experiment directory creation, saved config/metadata, CPU fallback behavior without requiring CUDA, and seed utility behavior. Keep tests fast and avoid GPU-dependent tests.

Documentation:
Update documentation only where implementation now makes planned architecture concrete. Update docs/ARCHITECTURE.md if actual module boundaries differ, docs/PLAN.md milestone status if appropriate, and README.md only if a small run command or project-structure note is useful. Do not expand documentation unnecessarily.

Prompt Book:
Append this exact development prompt, or a faithful preserved copy, to docs/PROMPTS_BOOK.md. Preserve previous entries, append only, maintain chronological order, identify the entry as Prompt 3 — Core Project Architecture, and do not rewrite Prompt 2.

Verification:
Before finishing, run the relevant pytest suite, run the infrastructure smoke command, inspect the created experiment output, verify configuration and metadata were persisted correctly, verify heavy/local artifacts remain ignored by Git, verify no dataset was downloaded, verify no model/training/evaluation/generation/GUI implementation was added, and inspect Git status.

You may commit the completed work with an appropriate commit message.

Stop after the core infrastructure is working. Do not begin the Fashion-MNIST data pipeline.
```

## Prompt 4 - Fashion-MNIST Data Pipeline

```text
PROMPT 4 — Fashion-MNIST Data Pipeline

Inspect the current repository before making any changes.

This task is ONLY to implement the Fashion-MNIST data pipeline.

Do NOT implement:
- Autoencoder or VAE models
- training loops
- evaluation logic
- generation
- GUI
- experiment comparison

Follow the existing project design and infrastructure in README.md, docs/PLAN.md, docs/ARCHITECTURE.md, configs/default.json, and existing core utilities under src/deep_learning_generative_models/.

Goal:
Implement a small, reliable, reusable Fashion-MNIST data layer that future Autoencoder and VAE prompts can use without duplication.

The dataset must download automatically when needed, cache locally, remain ignored by Git, work with the existing project paths/config structure, expose clean PyTorch DataLoaders, and remain fast enough for local development.

Inspect existing data/path infrastructure before implementing. Reuse existing path and config conventions. Do not create duplicate path/config systems.

Fashion-MNIST dataset module:
Create a focused data module using torchvision.datasets.FashionMNIST. It should build the training dataset, test dataset, training DataLoader, and test DataLoader. Dataset root must come from centralized project path utilities. Automatic download should occur when files are missing, and cached local files should be reused on subsequent runs. Do not hard-code machine-specific paths.

Transforms:
Use only transforms justified for Autoencoder/VAE image reconstruction. At minimum, convert images to PyTorch tensors. Do not add aggressive augmentation. Do not add normalization unless there is a clear reason compatible with later reconstruction/generation outputs and visualization can reverse it cleanly. Prefer the simplest valid approach.

DataLoader configuration:
Use the existing project configuration where appropriate. Batch size should come from config. Training loader should shuffle, test loader should not shuffle. Use conservative DataLoader settings that work reliably on Windows/PyCharm. Avoid multiprocessing complexity and premature optimization.

Data shapes and contracts:
Document and enforce the Fashion-MNIST tensor contract. Expected image shape is equivalent to [batch, 1, 28, 28]. Do not flatten images in the data pipeline; future convolutional models should receive image tensors with spatial dimensions preserved.

Optional small development subset:
Add a simple optional mechanism for limiting train/test samples during smoke/development runs. Full dataset remains the default. Subset behavior must be explicit and must not silently reduce the dataset. Avoid a second data pipeline. Minimal config keys such as train_subset_size and test_subset_size are acceptable, with null meaning full dataset.

Smoke / inspection command:
Add a minimal terminal command that loads configuration, initializes Fashion-MNIST loaders, retrieves one batch, and prints train dataset size, test dataset size, batch image shape, label shape, and value range or another useful sanity check. Do not start training. Reuse package/module execution style or create a small dedicated data inspection command if clearer.

Tests:
Add focused pytest coverage for expected tensor/image shape, channel count, batch size behavior, train/test shuffle intent where cleanly testable, subset configuration behavior, invalid subset sizes, and compatibility with project paths. Avoid tests that repeatedly download the real Fashion-MNIST dataset. Prefer mocking, temporary directories, or lightweight construction. A single integration/smoke verification against the real dataset is acceptable outside the unit-test suite. Keep tests fast.

Dependency discipline:
Use existing dependencies where possible. Do not add unnecessary libraries. torchvision is already part of the agreed stack.

Documentation:
Update documentation only where the data implementation now makes planned behavior concrete. At minimum, update docs/ARCHITECTURE.md with the actual data-layer boundary and tensor contract, docs/PLAN.md milestone status if appropriate, and README.md with only a short dataset/download note or data smoke command if useful.

Prompt Book:
Append this exact development prompt, or a faithful preserved copy, to docs/PROMPTS_BOOK.md. Append only, preserve all previous prompt entries, maintain chronological order, and identify this entry as Prompt 4 — Fashion-MNIST Data Pipeline. Do not rewrite earlier prompts.

Verification:
Before finishing, run the relevant pytest suite, run the data smoke/inspection command, verify Fashion-MNIST downloads automatically if not already cached, verify subsequent access uses the local cache, inspect one real batch, verify image tensor shape is [batch, 1, 28, 28], verify dataset files remain ignored by Git, verify no model/training/evaluation/generation/GUI code was added, and inspect Git status.

You may commit the completed work with an appropriate commit message.

Stop after the Fashion-MNIST data pipeline is working. Do not begin the Autoencoder implementation.
```
