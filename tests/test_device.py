from __future__ import annotations

import torch

from deep_learning_generative_models.device import get_device


def test_get_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device_info = get_device()

    assert device_info.device.type == "cpu"
    assert "CPU selected" in device_info.description
