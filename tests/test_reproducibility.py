from __future__ import annotations

import random

import torch

from deep_learning_generative_models.reproducibility import seed_everything


def test_seed_everything_repeats_python_and_torch_random_values() -> None:
    seed_everything(7)
    python_value = random.random()
    torch_value = torch.rand(1)

    report = seed_everything(7)

    assert random.random() == python_value
    assert torch.equal(torch.rand(1), torch_value)
    assert report.seed == 7
    assert report.deterministic_requested is True
