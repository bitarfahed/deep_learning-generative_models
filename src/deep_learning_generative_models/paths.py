"""Centralized project path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config_dir: Path
    default_config_file: Path
    data_dir: Path
    experiments_dir: Path
    outputs_dir: Path
    docs_dir: Path
    docs_results_dir: Path


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    raise RuntimeError("Could not find project root containing pyproject.toml")


@lru_cache(maxsize=1)
def get_project_paths() -> ProjectPaths:
    root = find_project_root()
    config_dir = root / "configs"
    docs_dir = root / "docs"
    return ProjectPaths(
        root=root,
        config_dir=config_dir,
        default_config_file=config_dir / "default.json",
        data_dir=root / "data" / "raw",
        experiments_dir=root / "experiments",
        outputs_dir=root / "outputs",
        docs_dir=docs_dir,
        docs_results_dir=docs_dir / "results",
    )
