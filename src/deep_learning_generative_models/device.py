"""Device selection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    description: str


def get_device() -> DeviceInfo:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
        return DeviceInfo(
            device=device,
            description=f"CUDA device selected: {device_name}",
        )

    return DeviceInfo(
        device=torch.device("cpu"),
        description="CPU selected; CUDA is not available through PyTorch.",
    )
