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
