from __future__ import annotations

from deep_learning_generative_models.paths import find_project_root, get_project_paths


def test_find_project_root_from_package_path() -> None:
    root = find_project_root()

    assert (root / "pyproject.toml").is_file()


def test_project_paths_are_relative_to_repo_root() -> None:
    paths = get_project_paths()

    assert paths.default_config_file == paths.root / "configs" / "default.json"
    assert paths.data_dir == paths.root / "data" / "raw"
    assert paths.experiments_dir == paths.root / "experiments"
    assert paths.outputs_dir == paths.root / "outputs"
    assert paths.docs_results_dir == paths.root / "docs" / "results"
