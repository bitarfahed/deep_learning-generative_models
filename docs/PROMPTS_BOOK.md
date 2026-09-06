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

## Prompt 5 - Convolutional Autoencoder Model

```text
PROMPT 5 — Convolutional Autoencoder Model

Inspect the current repository before making any changes.

This task is ONLY to implement the Convolutional Autoencoder model layer.

Do NOT implement:
- training loops
- optimizer logic
- experiment execution
- evaluation pipelines
- VAE logic
- random generation
- latent interpolation
- GUI

Reuse the existing project architecture, configuration system, Fashion-MNIST data contract, and package structure.

Goal:
Implement a clean Convolutional Autoencoder (AE) suitable for Fashion-MNIST.

The model must accept image tensors shaped [batch, 1, 28, 28], encode images into a latent representation, decode the latent representation back into reconstructed images, preserve the original image shape, and support fixed architecture presets small, medium, and deep.

The AE is the Deep Learning baseline for the project and will later be compared with the VAE.

Inspect existing contracts before implementing: configs/default.json, the Fashion-MNIST data module, docs/ARCHITECTURE.md, current config validation, and package naming/conventions. Do not duplicate existing abstractions.

Architecture presets:
Define three fixed model presets: small, medium, and deep. The user may select one preset but cannot arbitrarily edit its internal structure. Choose reasonable Fashion-MNIST preset definitions that increase progressively in capacity/depth, remain lightweight enough for local CPU development, are not unnecessarily large, and keep output reconstruction shape exactly 1 x 28 x 28. Prefer convolutional encoder blocks and a mirrored or logically corresponding decoder. Choose and document channel counts, kernel sizes, strides, and latent dimensions deliberately. Avoid residual networks, attention, transformers, pretrained models, unnecessary normalization layers, and architectural tricks without clear value.

Latent representation:
The Autoencoder should expose encode(x) -> z, decode(z) -> reconstruction, and forward(x) -> reconstruction. Use config latent_dim consistently where appropriate. Do not implement VAE statistics such as mu, logvar, sampling, or KL divergence.

Output range:
Choose a reconstruction output activation compatible with Fashion-MNIST preprocessing. Since inputs are in [0, 1], a sigmoid output is appropriate. Keep model and future reconstruction loss mathematically consistent. Do not change data preprocessing unless a real incompatibility is discovered.

Shape safety:
Input must be [batch, 1, 28, 28] and output must be [batch, 1, 28, 28] for all three presets. Avoid fragile hard-coded shape assumptions when a cleaner solution is available, but do not over-engineer dynamic shape handling for datasets outside the project scope.

Model factory:
Implement a small mechanism for constructing the selected AE preset from existing configuration. At this stage only AE needs to be constructible, while leaving the structure extensible enough for the VAE to be added later without duplicating preset-selection logic.

Human-readable architecture description:
Provide concise programmatic descriptions for each architecture because the future GUI will display them. Descriptions should reflect the actual implementation and should not expose mutable architecture parameters to the user.

Tests:
Add focused pytest tests for small, medium, and deep preset construction; forward pass shape; encode output shape; decode output shape; output value range if bounded activation is used; invalid architecture preset handling; and model factory behavior. Use small synthetic tensors. Do not require Fashion-MNIST download, GPU, training, or long-running operations.

Parameter counts:
Provide a lightweight utility or model property to report trainable parameter count. Use it in verification to compare Small / Medium / Deep. The expected relation should generally be small < medium < deep; reconsider preset design if that relation is not produced.

Documentation:
Update docs/ARCHITECTURE.md with actual AE encoder/decoder design, Small / Medium / Deep presets, latent representation, and input/output tensor contract. Update docs/PLAN.md milestone status if appropriate. Update README.md only with a concise model overview if useful. Do not describe training results because training has not been implemented.

Prompt Book:
Append this exact development prompt, or a faithful preserved copy, to docs/PROMPTS_BOOK.md. Append only, preserve previous entries, maintain chronological order, and identify this entry as Prompt 5 — Convolutional Autoencoder Model. Do not rewrite earlier prompts.

Verification:
Before finishing, run the full relevant pytest suite, instantiate Small/Medium/Deep AE presets, run synthetic forward passes through all three, verify input/output shapes, verify latent shapes, verify architecture descriptions match implementation, report trainable parameter counts for all presets, verify sensible capacity progression, verify no training/VAE/generation/GUI implementation was added, and inspect Git status.

You may commit the completed work with an appropriate commit message.

Stop after the Convolutional Autoencoder model layer is complete. Do not begin Autoencoder training.
```

## Prompt 6 - Autoencoder Training Pipeline

```text
PROMPT 6 — Autoencoder Training Pipeline

Inspect the current repository before making any changes.

This task is ONLY to implement the training pipeline for the existing Convolutional Autoencoder.

Do NOT implement:
- VAE
- random generation
- latent interpolation
- GUI
- AE vs VAE comparison
- unrelated evaluation features

Reuse the existing configuration system, device utilities, reproducibility utilities, experiment management, Fashion-MNIST data pipeline, and Autoencoder presets/factory.

Goal:
Implement a reusable Autoencoder training pipeline that can load configuration, select device, set random seeds, load Fashion-MNIST, construct the selected AE preset, train for the configured number of epochs, track training loss, save the trained checkpoint, persist training history/results in the experiment directory, and allow future runs to load the saved model instead of retraining automatically.

Keep the implementation simple and reusable for later VAE integration.

Inspect existing architecture before implementing: config.py, experiments.py, device.py, reproducibility.py, data pipeline, AE model/factory, and docs/ARCHITECTURE.md. Do not duplicate existing infrastructure.

Reconstruction loss:
Use an appropriate reconstruction loss consistent with Fashion-MNIST preprocessing and AE output range. Choose one clear default such as Binary Cross Entropy if model outputs are in [0, 1], or Mean Squared Error if better justified. Document the decision. Do not add multiple loss choices unless clearly needed.

Optimizer:
Use a simple default optimizer suitable for this project. Prefer Adam unless the repository already contains a justified alternative. Use configured learning rate. Do not introduce optimizer complexity or schedulers.

Training loop:
For each epoch, iterate over the training DataLoader, move images to device, zero gradients, forward pass, compute reconstruction loss, backward pass, optimizer step, and accumulate epoch loss. Report concise progress and avoid excessive logging. Keep the loop reusable enough for later VAE training without implementing VAE-specific logic.

Training entry point:
Provide a clean package-style training command conceptually similar to `uv run python -m deep_learning_generative_models.train --model ae`. It should load default config, allow at least model/preset selection, initialize the experiment, train the model, save outputs, and print the experiment location. Keep CLI arguments minimal.

Checkpoint saving:
Save a checkpoint in the experiment directory. Include enough information to reconstruct and load the model safely: model state_dict, model type, architecture preset, latent dimension/config needed by the model, relevant training configuration, and epoch count. Do not save the entire Python model object. Checkpoint files remain ignored by Git under artifact/experiment policy.

Training history:
Persist training history in a simple machine-readable format such as CSV or JSON. At minimum save epoch and training reconstruction loss. Do not add TensorBoard or database logging.

Runtime behavior:
The project must not retrain automatically whenever opened or imported. Training occurs only when the explicit training command is run. Saved checkpoints should be usable later by evaluation/GUI prompts.

Development-speed support:
Respect the project's short-run policy. The existing subset mechanism should remain usable for quick development/smoke training. Do not silently reduce the full dataset. A small explicit development run may use small architecture, limited subset, and 1-2 epochs. Do not add hidden fast-mode behavior.

Tests:
Add focused pytest coverage using synthetic tensors, tiny mocked/fake datasets, and very short training runs where appropriate. Test that one training step updates model parameters, loss is finite, training history is recorded, checkpoint is created, checkpoint contains required metadata/state, CPU training path works, and invalid/non-AE training requests fail cleanly. Do not download Fashion-MNIST inside unit tests, require CUDA, or run expensive training.

Smoke training verification:
Perform one real lightweight AE training smoke run using Fashion-MNIST, for example Small preset, small explicit training subset, and 1 epoch. Report selected device, dataset/subset size, training duration if easy to measure, final training loss, and checkpoint path. Do not perform a long/full training run.

Documentation:
Update docs/ARCHITECTURE.md with training flow and checkpoint contract, docs/PLAN.md milestone status if appropriate, and README.md with a concise training command and note that training is explicit. Do not add claims about model quality.

Prompt Book:
Append this exact development prompt, or a faithful preserved copy, to docs/PROMPTS_BOOK.md. Append only, preserve all previous entries, maintain chronological order, and identify this entry as Prompt 6 — Autoencoder Training Pipeline. Do not rewrite earlier prompts.

Verification:
Before finishing, run the relevant/full pytest suite, run the lightweight real AE smoke training, verify checkpoint creation, verify training-history persistence, inspect checkpoint metadata, verify experiment artifacts remain ignored by Git, verify training occurs only through explicit execution, verify no VAE/generation/GUI implementation was added, and inspect Git status.

You may commit the completed work with an appropriate commit message.

Stop after Autoencoder training works end-to-end. Do not begin Autoencoder evaluation/reconstruction output work.
```

## Prompt 7 - Autoencoder Evaluation & Reconstruction Outputs

```text
PROMPT 7 — Autoencoder Evaluation & Reconstruction Outputs

Inspect the current repository before making any changes.

This task is ONLY to implement evaluation and reconstruction outputs for the already implemented and trainable Autoencoder.

Do NOT implement:
- VAE
- random generation from latent space
- latent interpolation
- GUI
- AE vs VAE comparison
- new model architectures

Reuse the existing configuration system, device utilities, reproducibility utilities, experiment management, Fashion-MNIST data pipeline, Autoencoder presets/factory, checkpoint format, and training history/output conventions.

Goal:
Implement a clean evaluation workflow for trained AE checkpoints.

The system should load a trained AE checkpoint, reconstruct Fashion-MNIST images, compute reconstruction loss on evaluation/test data, save representative original-vs-reconstruction visual outputs, save a training-loss plot from persisted history, persist a concise evaluation summary, and avoid retraining the model.

Evaluation summary:
Persist a concise machine-readable evaluation summary using JSON or another existing simple project format. Include at minimum checkpoint identifier/path, model type, architecture preset, latent dimension, device used, number of evaluated samples, test reconstruction loss, and paths to generated evaluation artifacts where useful. Do not introduce a database or experiment-tracking framework.

Evaluation entry point:
Provide a package-style evaluation command conceptually similar to `uv run python -m deep_learning_generative_models.evaluate --checkpoint <path>`. The command should load the checkpoint, evaluate it, save outputs, and print a concise summary. Keep CLI arguments minimal. Do not add retraining behavior.

Output organization:
Reuse the existing experiment/artifact structure. Prefer storing evaluation outputs under the relevant experiment directory or a clearly associated subdirectory. Avoid scattering files across the repository. Representative local artifacts should remain ignored by Git unless intentionally copied later into portfolio documentation.

Tests:
Add focused pytest coverage for valid checkpoint loading, invalid/missing checkpoint handling, evaluation under torch.no_grad(), output reconstruction shape, finite evaluation loss, evaluation summary creation, reconstruction figure creation, training-loss plot creation, and CPU compatibility. Use synthetic/tiny data where practical. Do not require CUDA, perform long training, or repeatedly download Fashion-MNIST in unit tests.

Real smoke evaluation:
Use the lightweight AE checkpoint produced by the previous prompt if available. Run one real evaluation against Fashion-MNIST test data or a deliberate small evaluation subset. Report checkpoint used, selected device, number of evaluated samples, test reconstruction loss, reconstruction figure path, training-loss plot path, and evaluation summary path. Do not perform new training merely to improve the visual result. If the previous smoke checkpoint is weak, evaluate it honestly and note that a later full training run may improve reconstruction quality.

Documentation:
Update docs/ARCHITECTURE.md with evaluation/checkpoint-loading flow, docs/PLAN.md to mark AE evaluation/reconstruction complete, and README.md with a concise evaluation command and expected outputs if useful. Do not claim the AE is high quality merely because the pipeline works.

Prompt Book:
Append this exact development prompt, or a faithful preserved copy, to docs/PROMPTS_BOOK.md. Append only, preserve all previous entries, maintain chronological order, and identify this entry as Prompt 7 — Autoencoder Evaluation & Reconstruction Outputs. Do not rewrite earlier prompts.

Verification:
Before finishing, run the relevant/full pytest suite, run the real AE smoke evaluation, verify checkpoint loads without retraining, verify test reconstruction loss is finite, inspect the saved reconstruction figure, inspect the saved training-loss plot, inspect the evaluation summary, verify outputs are organized consistently, verify no VAE/generation/interpolation/GUI implementation was added, and inspect Git status.

You may commit the completed work with an appropriate commit message.

Stop after the Autoencoder baseline is fully evaluable and produces reconstruction outputs. Do not begin VAE implementation.
```

## Prompt 8 - Variational Autoencoder Model

```text
PROMPT 8 — Variational Autoencoder Model

Inspect the current repository before making any changes.

This task is ONLY to implement the Variational Autoencoder model layer.

Do NOT implement:
- VAE training loop
- generation CLI
- latent interpolation workflow
- AE vs VAE experiments
- GUI
- new dataset logic

Reuse the existing package structure, configuration system, Fashion-MNIST tensor contract, Autoencoder preset conventions, model factory patterns, and documentation conventions.

Goal:
Implement a clean VAE model for Fashion-MNIST that is structurally aligned with the existing Convolutional Autoencoder baseline.

The VAE must accept [batch, 1, 28, 28], encode images into latent-distribution parameters, sample a latent vector using the reparameterization trick, decode latent vectors back into [batch, 1, 28, 28], support the fixed Small / Medium / Deep architecture presets, and expose the components needed later for training, reconstruction, generation, interpolation, and GUI exploration.

Inspect the existing AE architecture before implementing: Autoencoder implementation, preset definitions, model factory, config validation, and docs/ARCHITECTURE.md. Reuse architectural conventions where sensible. The VAE should feel like a natural extension of the AE, not a separate system. Do not duplicate preset-selection logic unnecessarily.

VAE encoder:
Implement an encoder that produces mu and logvar for the latent distribution. The encoder should reuse or closely follow the convolutional feature-extraction pattern already established by the AE presets. Keep the architecture simple and explainable. Do not add attention, residual blocks, pretrained backbones, unnecessary normalization, or unrelated architectural complexity.

Reparameterization:
Implement a dedicated, testable reparameterization function: std = exp(0.5 * logvar), eps = random normal noise, z = mu + eps * std. It must be training-compatible, differentiable, numerically reasonable, and isolated enough to test. Do not detach tensors in a way that breaks gradient flow.

Decoder:
Implement a decoder that maps latent vectors back to image space: [batch, latent_dim] -> [batch, 1, 28, 28]. The output activation must remain compatible with current Fashion-MNIST preprocessing and reconstruction-loss policy. Reuse decoder conventions from the AE where appropriate.

Forward interface:
Expose encode(x) -> mu, logvar; reparameterize(mu, logvar) -> z; decode(z) -> reconstruction; and forward(x) -> reconstruction, mu, logvar, z. A small typed container/dataclass may be used if it improves clarity. Future training code must be able to access reconstruction, mu, logvar, and z.

Fixed architecture presets:
Support small, medium, and deep. Maintain the same philosophy as the AE: fixed internal architectures, progressively increasing capacity, lightweight enough for Fashion-MNIST, and no arbitrary user modification of internal layers. Prefer consistency with the AE preset family. Document any justified differences.

Latent dimension:
Use the existing configuration system for latent_dim. Keep latent_dim explicit and compatible with future random generation, latent interpolation, and latent sliders in the GUI. Do not hard-code a single latent size. Validate latent dimension appropriately.

Model factory integration:
Extend existing model construction/factory so model type ae creates the Autoencoder and model type vae creates the Variational Autoencoder. Avoid parallel factories if one clean shared mechanism is sufficient. Do not break existing AE behavior.

Human-readable architecture descriptions:
Extend the architecture-description mechanism so the future GUI can display concise descriptions for VAE presets. Descriptions should reflect the implementation, mentioning convolutional encoder depth, latent dimension/distribution, and decoder capacity. Keep descriptions concise.

Parameter counts:
Report trainable parameter counts for VAE Small, Medium, and Deep. Presets should represent a sensible capacity progression. Do not optimize merely for parameter count.

Tests:
Add focused pytest coverage for Small/Medium/Deep construction, forward output shapes, mu shape, logvar shape, latent z shape, decoder output shape, output value range, reparameterization output shape, gradient flow through reparameterization, factory construction using model type vae, invalid model/preset handling, and AE factory behavior remaining intact. Use synthetic tensors only. Do not download Fashion-MNIST, require CUDA, or train the VAE.

Documentation:
Update docs/ARCHITECTURE.md with VAE encoder, mu/logvar, reparameterization, latent z, decoder, Small/Medium/Deep preset relationship, and forward data flow. Update docs/PLAN.md to mark VAE model complete if appropriate. Update README.md only with a concise VAE architecture note if useful. Do not document training results because VAE training is not implemented yet.

Prompt Book:
Append this exact development prompt, or a faithful preserved copy, to docs/PROMPTS_BOOK.md. Append only, preserve all previous entries, maintain chronological order, and identify this entry as Prompt 8 — Variational Autoencoder Model. Do not rewrite earlier prompts.

Verification:
Before finishing, run the full relevant pytest suite, instantiate AE and VAE Small/Medium/Deep presets, run synthetic forward passes, verify all VAE tensor shapes, verify reparameterization remains differentiable, verify output range contract, report VAE parameter counts, verify AE behavior/tests remain valid, verify no VAE training/generation/interpolation/GUI implementation was added, and inspect Git status.

You may commit the completed work with an appropriate commit message.

Stop after the VAE model layer is complete. Do not begin VAE training.
```

## Prompt 9 - VAE Training Integration

```text
PROMPT 9 — VAE Training Integration

Inspect the existing repository and follow the established architecture and documentation.

Goal:
Integrate VAE training into the existing training pipeline without duplicating the AE training system.

Requirements:

1. Extend the current shared training pipeline to support:

   - AE
   - VAE

2. For VAE training implement the standard loss:

   - reconstruction loss
   - KL-divergence loss
   - total VAE loss

Keep the loss implementation explicit and easy to understand.

3. Persist VAE training history including:

   - total loss
   - reconstruction loss
   - KL loss

4. Use the existing:

   - configuration
   - device selection
   - seeds
   - Fashion-MNIST pipeline
   - experiment directories
   - checkpoint system

Do not create parallel infrastructure unnecessarily.

5. VAE checkpoints must contain enough metadata to reconstruct and load the correct model later.

6. Training must remain explicit. Opening/importing the project must never trigger training.

7. Add focused, fast tests for:

   - VAE loss
   - VAE training step
   - checkpoint creation/loading metadata
   - persisted loss history
   - regression of existing AE training behavior

Do not require CUDA or long dataset training in unit tests.

8. Run one lightweight real VAE smoke training using Fashion-MNIST, preferably Small + small subset + 1 epoch.

Do not perform a long/full training run yet.

9. Update README/PLAN/ARCHITECTURE only where the implemented behavior makes an update necessary.

10. Append this prompt to docs/PROMPTS_BOOK.md as:

Prompt 9 — VAE Training Integration

Preserve all previous entries.

Do NOT implement yet:

- random generation workflow
- latent interpolation
- AE vs VAE experiments
- GUI

Run the relevant/full test suite and commit the completed work.

Final report should be short and contain only:

- main changes
- test results
- VAE smoke-training result
- any problem, design deviation, or decision requiring attention

Stop after VAE training works end-to-end.
```

## Prompt 10 - Generation & Latent-Space Exploration Core

```text
PROMPT 10 — Generation & Latent-Space Exploration Core

Inspect the repository and follow the existing architecture/documentation.

Goal:
Add the core VAE generation and latent-space exploration functionality, without GUI.

Requirements:

1. Implement reliable loading of a trained VAE checkpoint without retraining.

2. Implement random generation:

   - sample latent vectors from the VAE prior
   - decode them into Fashion-MNIST images
   - save a grid of generated images

3. Implement latent interpolation:

   - encode two selected test images
   - use their latent representations
   - interpolate between them across several steps
   - decode and save the interpolation sequence

4. Add reusable APIs that the future GUI can call directly.
   Keep generation logic independent from UI code.

5. Provide a small package-style CLI for running generation/interpolation from an existing checkpoint.

6. Save outputs inside the existing experiment/artifact structure.

7. Add focused, fast tests for:

   - random latent sampling
   - generated image shapes/ranges
   - interpolation endpoints and intermediate shapes
   - checkpoint loading
   - deterministic behavior where a fixed seed makes it practical

8. Run a lightweight real smoke test using an existing VAE checkpoint.
   Do not perform new training unless absolutely necessary for verification.

9. Update documentation only where necessary.

10. Append this prompt to docs/PROMPTS_BOOK.md as:

Prompt 10 — Generation & Latent-Space Exploration Core

Preserve all previous entries.

Do NOT implement:

- GUI
- AE vs VAE experiment suite
- long/full training
- GAN/diffusion or other generative models

Run the relevant/full test suite and commit the completed work.

Final report should contain only:

- main changes
- tests/results
- generation/interpolation smoke result
- any problem or decision requiring attention

Stop after generation and latent interpolation work without GUI.
```

## Prompt 11 - AE vs VAE Experiments & Comparison

```text
PROMPT 11 — AE vs VAE Experiments & Comparison

Inspect the repository and follow the existing architecture/documentation.

Goal:
Add a small, reproducible comparison workflow between the existing AE and VAE.

Requirements:

1. Compare AE and VAE under reasonably comparable conditions:

   - same Fashion-MNIST data policy
   - same architecture preset where practical
   - same seed
   - comparable training settings

2. Compare only what is meaningful for this project:

   - test reconstruction loss
   - qualitative reconstruction examples
   - training-loss behavior
   - VAE generation capability

Do not add unnecessary metrics.

3. Produce a concise experiment summary containing:
   - configurations used
   - AE reconstruction loss
   - VAE reconstruction loss
   - links/paths to representative outputs
   - a short factual comparison

Do not automatically claim one model is "better"; explain the reconstruction-vs-generative tradeoff shown by the results.

4. Reuse existing training, evaluation, checkpoint, and artifact infrastructure.
   Do not duplicate pipelines.

5. Keep experiment runs configurable and reproducible.
   Do not require long training as part of normal execution.

6. Add focused tests for the comparison/result aggregation logic.
   Do not test model quality in unit tests.

7. Perform a lightweight end-to-end comparison using available checkpoints where possible.
   Do not retrain models unnecessarily.

8. Update documentation only where necessary.

9. Append this prompt to docs/PROMPTS_BOOK.md as:

Prompt 11 — AE vs VAE Experiments & Comparison

Preserve all previous entries.

Do NOT implement:

- GUI
- new model families
- hyperparameter search
- large experiment suites

Run the relevant/full test suite and commit the completed work.

Final report should contain only:

- main changes
- tests/results
- comparison result
- any problem or decision requiring attention

Stop after the AE vs VAE comparison workflow works.
```

## Prompt 12 - Core Engineering & QA Audit

```text
PROMPT 12 — Core Engineering & QA Audit

Inspect the complete current repository before making changes.

Goal:
Audit and stabilize the completed non-GUI core before GUI development begins.

This is primarily a QA task, not a feature-development task.

## Audit

Verify the complete workflow:

Fashion-MNIST
→ AE/VAE construction
→ training
→ checkpoint saving
→ checkpoint loading
→ evaluation
→ VAE generation/interpolation
→ AE vs VAE comparison

Pay particular attention to:

1. Training

- Check AE/VAE shared training logic for duplication, obsolete paths, or behavioral drift.
- Review the relationship between existing training functions such as train_one_epoch and train_one_epoch_with_metrics.
- Remove dead/obsolete duplication only if clearly safe.
- Verify AE behavior is not changed accidentally by VAE integration.

2. VAE loss

- Verify reconstruction + KL loss is mathematically and numerically coherent with the current reduction/scaling policy.
- Do not introduce beta-VAE or new loss features unless fixing an actual correctness issue.
- Document any important finding.

3. Checkpoints

- Audit save/load behavior across training, evaluation, generation, and comparison.
- Verify required metadata is consistent and validated.
- Avoid duplicated checkpoint-loading logic where a small safe consolidation is clearly justified.

4. Models

- Verify Small/Medium/Deep presets form a sensible capacity progression for both AE and VAE.
- Verify all input/output/latent shape contracts.
- Verify architecture descriptions match implementation.

5. Reproducibility and devices

- Verify seed handling.
- Verify CPU operation end-to-end.
- Verify CUDA detection/fallback remains safe.
- Report the current CUDA/PyTorch status, but do not reinstall PyTorch, CUDA, or drivers.

6. Artifacts and Git

- Verify datasets, checkpoints, experiments, caches, IDE files, and virtual environments are not accidentally tracked.
- Do not delete useful local artifacts unnecessarily.

7. Tests
   Audit the existing tests for meaningful coverage, not only shape/smoke checks.

Add regression tests only where a real coverage gap or bug risk is identified.

Keep tests fast and independent of CUDA or long training.

8. Documentation
   Verify README.md, docs/PLAN.md, and docs/ARCHITECTURE.md accurately describe the implemented system.

Correct factual inconsistencies if found.
Do not expand documentation unnecessarily.

9. Future GUI boundary
   Verify that reconstruction, checkpoint loading, VAE generation, latent handling, and interpolation are reusable without coupling them to CLI internals.

Only make small refactors if necessary to establish a clean callable boundary for the future GUI.

Do NOT implement GUI code.

## Scope Discipline

Fix:

- actual bugs
- correctness problems
- clear dead/obsolete code
- unsafe duplication
- important missing regression coverage
- documentation inconsistencies

Do NOT:

- redesign working architecture without need
- add features
- add new model families
- perform hyperparameter optimization
- add TensorBoard/cloud/database/API infrastructure
- implement GUI
- perform long training runs

Prefer leaving correct working code unchanged.

## Verification

Run the full test suite.

Perform lightweight end-to-end verification of the major existing workflows where practical using existing checkpoints/artifacts.

Do not retrain models unnecessarily.

Inspect final Git status and tracked files.

## Prompt Book

Append this prompt to docs/PROMPTS_BOOK.md as:

Prompt 12 — Core Engineering & QA Audit

Preserve all previous entries.

## Commit

If changes are justified, commit them with an appropriate message.

If the audit finds that no code changes are needed, do not manufacture changes merely to create a commit; documentation/Prompt Book updates may still be committed as appropriate.

## Final Report

Keep the report concise.

Include only:

- bugs/problems found
- changes made
- tests and verification results
- CUDA/PyTorch device status
- whether the core is ready for GUI development
- any issue requiring my decision

Stop after the core QA/stabilization pass.

Do not begin GUI implementation.
```

## Prompt 13 - Interactive GUI Core

```text
PROMPT 13 — Interactive GUI Core

Inspect the repository and reuse the existing stable core APIs.

Goal:
Add a simple single-window Tkinter GUI for exploring already-trained AE/VAE models.

The GUI is an educational visualization layer, not the main project.

Requirements:

1. Use Tkinter.
   Keep the interface simple and functional.
   Do not introduce another GUI framework.

2. Allow the user to:

- choose model type: AE or VAE
- choose Small / Medium / Deep preset
- see a short description of the selected architecture
- select/load a compatible existing checkpoint
- browse a small set of Fashion-MNIST test images
- select an image
- run reconstruction
- view Original vs Reconstruction

3. For VAE checkpoints, also provide:

- Generate Random
- display the generated image(s)

Reuse existing checkpoint, reconstruction, generation, device, and data APIs.
Do not duplicate ML logic inside GUI code.

4. The GUI must never train automatically.

Do not add training controls in this prompt.

5. Handle basic errors clearly:

- missing checkpoint
- incompatible checkpoint/model/preset
- missing dataset
- invalid selection

The GUI should remain usable after recoverable errors.

6. Keep ML work outside the Tkinter UI layer where possible.
   The GUI should orchestrate existing services/functions rather than reimplement model behavior.

7. Add focused tests for non-visual GUI/controller logic where practical.
   Do not create fragile pixel/layout tests.

8. Perform a manual smoke verification:

- launch GUI
- load an existing checkpoint
- select a Fashion-MNIST image
- reconstruct it
- verify Original/Reconstruction display
- verify VAE random generation

9. Update documentation only where necessary.

10. Append this prompt to docs/PROMPTS_BOOK.md as:

Prompt 13 — Interactive GUI Core

Preserve all previous entries.

Do NOT implement yet:

- latent sliders
- interpolation UI
- training from GUI
- complex navigation/tabs
- visual polish beyond basic usability

Run the full test suite and commit the completed work.

Final report should contain only:

- main changes
- tests/results
- GUI smoke-test result
- any problem or decision requiring attention

Stop after the basic GUI works.
```
