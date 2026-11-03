"""Residual CNN that refines the interpolated LS channel estimate."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .config import ModelConfig


class ResidualCNN(torch.nn.Module):
    """Small fully convolutional network that predicts a correction to its input.

    Input (B, 3, S, K): Re and Im of the LS + linear estimate plus the SNR plane, as built
    by nn.dataset.to_input. Output (B, 2, S, K): Re and Im of the refined estimate.

    The last layer starts at zero, so an untrained model returns the LS + linear estimate
    unchanged and training only has to learn the improvement.
    """

    def __init__(
        self, channels: int = 16, layers: int = 4, kernel_size: int = 3, use_snr: bool = True
    ) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("need at least two layers")
        self.use_snr = use_snr
        pad = kernel_size // 2  # "same" padding keeps the 14 x 72 grid size
        in_channels = 3 if use_snr else 2
        blocks: list[torch.nn.Module] = []
        for _ in range(layers - 1):
            blocks += [
                torch.nn.Conv2d(in_channels, channels, kernel_size, padding=pad),
                torch.nn.ReLU(),
            ]
            in_channels = channels
        last = torch.nn.Conv2d(channels, 2, kernel_size, padding=pad)
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        self.body = torch.nn.Sequential(*blocks, last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x if self.use_snr else x[:, :2]
        return x[:, :2] + self.body(features)


def build_model(cfg: ModelConfig) -> ResidualCNN:
    return ResidualCNN(cfg.channels, cfg.layers, cfg.kernel_size, cfg.use_snr)


def save_model(model: ResidualCNN, cfg: ModelConfig, path: str | Path) -> None:
    """Store weights together with the architecture, so load_model needs nothing else."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": asdict(cfg), "state_dict": model.state_dict()}, path)


def load_model(path: str | Path) -> ResidualCNN:
    checkpoint = torch.load(path, weights_only=True)
    model = build_model(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def macs_per_slot(model: torch.nn.Module, n_symbols: int = 14, n_subcarriers: int = 72) -> int:
    """Multiply-accumulate operations for one slot (convolutions only, "same" padding)."""
    macs = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            kh, kw = m.kernel_size
            macs += m.out_channels * m.in_channels * kh * kw * n_symbols * n_subcarriers
    return macs
