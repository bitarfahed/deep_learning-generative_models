"""Configuration loading and validation for experiment infrastructure."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from deep_learning_generative_models.paths import get_project_paths

ModelType = Literal["ae", "vae"]
ArchitecturePreset = Literal["small", "medium", "deep"]

VALID_MODEL_TYPES = {"ae", "vae"}
VALID_ARCHITECTURE_PRESETS = {"small", "medium", "deep"}


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved experiment configuration shared by future project stages."""

    model_type: ModelType
    architecture_preset: ArchitecturePreset
    epochs: int
    batch_size: int
    learning_rate: float
    latent_dim: int
    random_seed: int
    train_subset_size: int | None = None
    test_subset_size: int | None = None

    def validate(self) -> None:
        if self.model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model_type: {self.model_type!r}")
        if self.architecture_preset not in VALID_ARCHITECTURE_PRESETS:
            raise ValueError(
                f"Invalid architecture_preset: {self.architecture_preset!r}"
            )
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.train_subset_size is not None and self.train_subset_size <= 0:
            raise ValueError("train_subset_size must be positive or null")
        if self.test_subset_size is not None and self.test_subset_size <= 0:
            raise ValueError("test_subset_size must be positive or null")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_from_dict(values: dict[str, Any]) -> ExperimentConfig:
    required = {
        "model_type",
        "architecture_preset",
        "epochs",
        "batch_size",
        "learning_rate",
        "latent_dim",
        "random_seed",
    }
    missing = required.difference(values)
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")

    config = ExperimentConfig(
        model_type=values["model_type"],
        architecture_preset=values["architecture_preset"],
        epochs=int(values["epochs"]),
        batch_size=int(values["batch_size"]),
        learning_rate=float(values["learning_rate"]),
        latent_dim=int(values["latent_dim"]),
        random_seed=int(values["random_seed"]),
        train_subset_size=_optional_positive_int(values.get("train_subset_size")),
        test_subset_size=_optional_positive_int(values.get("test_subset_size")),
    )
    config.validate()
    return config


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def load_config(path: Path | str) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        values = json.load(file)
    if not isinstance(values, dict):
        raise ValueError("Configuration file must contain a JSON object")
    return config_from_dict(values)


def load_default_config() -> ExperimentConfig:
    return load_config(get_project_paths().default_config_file)


def save_config(config: ExperimentConfig, path: Path | str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(config.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")
