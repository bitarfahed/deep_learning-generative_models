"""Convolutional Autoencoder model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import torch
from torch import Tensor, nn

from deep_learning_generative_models.config import ArchitecturePreset, ExperimentConfig
from deep_learning_generative_models.data import IMAGE_SHAPE

ModelKind = Literal["ae"]

_ENCODED_SPATIAL_SIZE: Final[int] = 7


@dataclass(frozen=True)
class AutoencoderPreset:
    name: ArchitecturePreset
    encoder_channels: tuple[int, ...]
    description: str

    @property
    def encoded_channels(self) -> int:
        return self.encoder_channels[-1]

    @property
    def conv_block_count(self) -> int:
        return len(self.encoder_channels)


AUTOENCODER_PRESETS: Final[dict[str, AutoencoderPreset]] = {
    "small": AutoencoderPreset(
        name="small",
        encoder_channels=(16, 32),
        description=(
            "Small: 2 convolutional encoder blocks, compact latent "
            "representation, lightweight reconstruction baseline."
        ),
    ),
    "medium": AutoencoderPreset(
        name="medium",
        encoder_channels=(16, 32, 64),
        description=(
            "Medium: 3 convolutional encoder blocks, increased channel "
            "capacity, balanced default for Fashion-MNIST reconstruction."
        ),
    ),
    "deep": AutoencoderPreset(
        name="deep",
        encoder_channels=(32, 64, 128, 128),
        description=(
            "Deep: 4 convolutional encoder blocks with higher channel "
            "capacity for a stronger convolutional reconstruction baseline."
        ),
    ),
}


class ConvolutionalAutoencoder(nn.Module):
    """Convolutional Autoencoder for Fashion-MNIST images."""

    def __init__(self, preset: AutoencoderPreset, latent_dim: int) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.preset = preset
        self.latent_dim = latent_dim
        self.encoder = _build_encoder(preset.encoder_channels)
        flattened_dim = preset.encoded_channels * _ENCODED_SPATIAL_SIZE**2
        self.to_latent = nn.Linear(flattened_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flattened_dim)
        self.decoder = _build_decoder(preset.encoder_channels)

    @property
    def description(self) -> str:
        return self.preset.description

    @property
    def trainable_parameter_count(self) -> int:
        return count_trainable_parameters(self)

    def encode(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        return self.to_latent(torch.flatten(features, start_dim=1))

    def decode(self, z: Tensor) -> Tensor:
        features = self.from_latent(z)
        features = features.view(
            z.shape[0],
            self.preset.encoded_channels,
            _ENCODED_SPATIAL_SIZE,
            _ENCODED_SPATIAL_SIZE,
        )
        return self.decoder(features)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def build_autoencoder(
    architecture_preset: ArchitecturePreset | str,
    latent_dim: int,
) -> ConvolutionalAutoencoder:
    preset = get_autoencoder_preset(architecture_preset)
    return ConvolutionalAutoencoder(preset=preset, latent_dim=latent_dim)


def build_model(config: ExperimentConfig) -> ConvolutionalAutoencoder:
    if config.model_type != "ae":
        raise ValueError("Only the AE model is implemented at this milestone")
    return build_autoencoder(
        architecture_preset=config.architecture_preset,
        latent_dim=config.latent_dim,
    )


def get_autoencoder_preset(
    architecture_preset: ArchitecturePreset | str,
) -> AutoencoderPreset:
    try:
        return AUTOENCODER_PRESETS[architecture_preset]
    except KeyError as error:
        raise ValueError(f"Invalid AE architecture preset: {architecture_preset!r}") from error


def get_autoencoder_description(architecture_preset: ArchitecturePreset | str) -> str:
    return get_autoencoder_preset(architecture_preset).description


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def _build_encoder(channels: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = IMAGE_SHAPE[0]
    for index, out_channels in enumerate(channels):
        stride = 2 if index < 2 else 1
        layers.extend(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                ),
                nn.ReLU(),
            ]
        )
        in_channels = out_channels
    return nn.Sequential(*layers)


def _build_decoder(channels: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_channels = channels[-1]

    for out_channels in reversed(channels[1:-1]):
        layers.extend(
            [
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(),
            ]
        )
        current_channels = out_channels

    first_channel = channels[0]
    layers.extend(
        [
            nn.ConvTranspose2d(
                in_channels=current_channels,
                out_channels=first_channel,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=first_channel,
                out_channels=IMAGE_SHAPE[0],
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        ]
    )
    return nn.Sequential(*layers)
