"""Minimal smoke entry point for core infrastructure."""

from __future__ import annotations

import argparse

from deep_learning_generative_models.config import load_config, load_default_config
from deep_learning_generative_models.device import get_device
from deep_learning_generative_models.experiments import create_experiment
from deep_learning_generative_models.reproducibility import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Core infrastructure smoke command. This does not train a model."
    )
    parser.add_argument(
        "--config",
        help="Optional path to a JSON config file. Defaults to configs/default.json.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config) if args.config else load_default_config()
    device_info = get_device()
    seed_report = seed_everything(config.random_seed)
    experiment = create_experiment(config, device_info)

    print("Core infrastructure smoke check complete.")
    print(f"Model type: {config.model_type}")
    print(f"Architecture preset: {config.architecture_preset}")
    print(f"Device: {device_info.description}")
    print(f"Seed: {seed_report.seed}")
    print(f"Experiment directory: {experiment.path}")
    print(f"Saved config: {experiment.config_path}")
    print(f"Saved metadata: {experiment.metadata_path}")


if __name__ == "__main__":
    main()
