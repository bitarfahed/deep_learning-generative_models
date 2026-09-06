"""Experiment directory creation and metadata persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from deep_learning_generative_models.config import ExperimentConfig, save_config
from deep_learning_generative_models.device import DeviceInfo
from deep_learning_generative_models.paths import ProjectPaths, get_project_paths

DATASET_SLUG = "fashion-mnist"


@dataclass(frozen=True)
class ExperimentMetadata:
    experiment_id: str
    experiment_name: str
    created_at_utc: str
    model_type: str
    architecture_preset: str
    random_seed: int
    selected_device: str
    device_description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentRecord:
    path: Path
    config_path: Path
    metadata_path: Path
    metadata: ExperimentMetadata


def artifact_prefix(config: ExperimentConfig) -> str:
    return f"{config.model_type}-{config.architecture_preset}"


def checkpoint_filename(config: ExperimentConfig) -> str:
    return f"{artifact_prefix(config)}-checkpoint.pt"


def training_history_filename(config: ExperimentConfig) -> str:
    return f"{artifact_prefix(config)}-training-history.csv"


def config_filename(config: ExperimentConfig) -> str:
    return f"{artifact_prefix(config)}-config.json"


def metadata_filename(config: ExperimentConfig) -> str:
    return f"{artifact_prefix(config)}-metadata.json"


def experiment_directory_name(
    config: ExperimentConfig,
    created_at: datetime,
    experiment_id: str,
) -> str:
    return (
        f"{config.model_type}-{config.architecture_preset}-{DATASET_SLUG}-"
        f"{config.epochs}ep-{created_at:%Y%m%d-%H%M%S}-{experiment_id}"
    )


def create_experiment(
    config: ExperimentConfig,
    device_info: DeviceInfo,
    paths: ProjectPaths | None = None,
) -> ExperimentRecord:
    project_paths = paths or get_project_paths()
    created_at = datetime.now(UTC)
    experiment_id = uuid4().hex[:8]
    experiment_name = experiment_directory_name(config, created_at, experiment_id)
    experiment_path = project_paths.experiments_dir / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=False)

    config_path = experiment_path / config_filename(config)
    metadata_path = experiment_path / metadata_filename(config)
    save_config(config, config_path)

    metadata = ExperimentMetadata(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        created_at_utc=created_at.isoformat(),
        model_type=config.model_type,
        architecture_preset=config.architecture_preset,
        random_seed=config.random_seed,
        selected_device=str(device_info.device),
        device_description=device_info.description,
    )
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")

    return ExperimentRecord(
        path=experiment_path,
        config_path=config_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )
