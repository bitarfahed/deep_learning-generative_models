"""Reproducibility helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SeedReport:
    seed: int
    cuda_seeded: bool
    deterministic_requested: bool


def seed_everything(seed: int, deterministic: bool = True) -> SeedReport:
    random.seed(seed)
    torch.manual_seed(seed)

    cuda_seeded = torch.cuda.is_available()
    if cuda_seeded:
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    return SeedReport(
        seed=seed,
        cuda_seeded=cuda_seeded,
        deterministic_requested=deterministic,
    )
