from __future__ import annotations

import json

import pytest

from deep_learning_generative_models.config import (
    config_from_dict,
    load_default_config,
    load_config,
)


def test_load_default_config() -> None:
    config = load_default_config()

    assert config.model_type == "ae"
    assert config.architecture_preset == "small"
    assert config.epochs > 0
    assert config.batch_size > 0
    assert config.learning_rate > 0
    assert config.latent_dim > 0
    assert isinstance(config.random_seed, int)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "gan"),
        ("architecture_preset", "wide"),
        ("epochs", 0),
        ("batch_size", 0),
        ("learning_rate", 0),
        ("latent_dim", 0),
        ("train_subset_size", 0),
        ("test_subset_size", 0),
    ],
)
def test_invalid_config_values_are_rejected(field: str, value: object) -> None:
    values = {
        "model_type": "ae",
        "architecture_preset": "small",
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.001,
        "latent_dim": 16,
        "random_seed": 42,
        "train_subset_size": None,
        "test_subset_size": None,
    }
    values[field] = value

    with pytest.raises(ValueError):
        config_from_dict(values)


def test_subset_sizes_are_optional() -> None:
    config = config_from_dict(
        {
            "model_type": "vae",
            "architecture_preset": "deep",
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.001,
            "latent_dim": 8,
            "random_seed": 99,
        }
    )

    assert config.train_subset_size is None
    assert config.test_subset_size is None


def test_load_config_requires_json_object(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_config(path)
