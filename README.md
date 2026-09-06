# deep_learning-generative_models

Build an interactive portfolio project demonstrating both Deep Learning and Generative Modeling through image reconstruction, latent representations, and generation.

The project emphasizes educational value, clear architecture, reproducible experiments, and understandable model behavior rather than visually impressive generated images.

## Planned progression

1. Repository bootstrap
2. Project design
3. Core project architecture
4. Fashion-MNIST data pipeline
5. Convolutional Autoencoder baseline
6. Autoencoder training and reconstruction evaluation
7. Variational Autoencoder (VAE)
8. VAE training
9. Latent-space sampling and interpolation
10. AE vs VAE comparison
11. Engineering/QA and stabilization
12. Interactive GUI for exploring trained models and latent space
13. Portfolio closure

## Agreed direction

- Technology: Python, PyTorch, torchvision, matplotlib, and pytest.
- Dataset: Fashion-MNIST, downloaded automatically when needed and cached locally.
- Data contract: image batches preserve spatial dimensions as `[batch, 1, 28, 28]`.
- Baseline: Convolutional Autoencoder (AE) for image representation and reconstruction.
- Generative model: Variational Autoencoder (VAE) for reconstruction training, latent distributions, and later generation/interpolation.
- AE architecture presets: Small, Medium, and Deep fixed convolutional presets.

## Current status

Repository bootstrap, project design documentation, core infrastructure, the Fashion-MNIST data pipeline, the Convolutional Autoencoder model layer, explicit Autoencoder training, AE reconstruction evaluation, the VAE model layer, and explicit VAE training are complete. Generation utilities, latent interpolation workflows, and GUI code have not been implemented yet.

Run the infrastructure smoke check from the source tree:

```bash
uv run python -m deep_learning_generative_models
```

Inspect the Fashion-MNIST data pipeline:

```bash
uv run python -m deep_learning_generative_models.data_inspect
```

Run an explicit lightweight Autoencoder training smoke run:

```bash
uv run python -m deep_learning_generative_models.train --model ae --preset small --epochs 1 --train-subset-size 256 --test-subset-size 64
```

Run an explicit lightweight VAE training smoke run:

```bash
uv run python -m deep_learning_generative_models.train --model vae --preset small --epochs 1 --train-subset-size 256 --test-subset-size 64
```

Training only runs when the training command is executed.

Evaluate a trained Autoencoder checkpoint:

```bash
uv run python -m deep_learning_generative_models.evaluate --checkpoint experiments/<experiment-name>/checkpoint.pt --max-samples 128
```

Evaluation loads an existing checkpoint and writes reconstruction figures, a training-loss plot, and a JSON summary under the same experiment directory.

The future GUI is intended as an educational exploration layer over working models, not as the main purpose of the project. Training must not run automatically just because the GUI/application is opened.

Cloud deployment, authentication, databases, backend APIs, LLM integration, TensorFlow, GANs, diffusion models, transformers, unrelated AI features, and unnecessary infrastructure are outside the current scope unless later justified.

## AI assistance disclosure

This project is being developed with significant assistance from ChatGPT and Codex for planning, documentation, code generation, review, and repository maintenance. That assistance should not be interpreted as equivalent to the author's independent implementation ability.

## Project layout

```text
src/deep_learning_generative_models/
  config.py
  device.py
  experiments.py
  paths.py
  reproducibility.py
  data.py
  data_inspect.py
  models.py
  train.py
  evaluate.py
tests/
configs/
  default.json
docs/
  PLAN.md
  ARCHITECTURE.md
  PROMPTS_BOOK.md
```
