"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OFDMConfig:
    n_fft: int = 128
    n_subcarriers: int = 72
    n_symbols: int = 14
    cp_length: int = 9
    subcarrier_spacing_hz: float = 15e3
    modulation: str = "qpsk"

    @property
    def sample_rate_hz(self) -> float:
        return self.n_fft * self.subcarrier_spacing_hz

    @property
    def sample_period_s(self) -> float:
        return 1.0 / self.sample_rate_hz

    @property
    def symbol_length(self) -> int:
        """Samples per OFDM symbol including the cyclic prefix."""
        return self.n_fft + self.cp_length

    @property
    def slot_length(self) -> int:
        return self.n_symbols * self.symbol_length

    @property
    def symbol_duration_s(self) -> float:
        return self.symbol_length * self.sample_period_s


@dataclass(frozen=True)
class PilotConfig:
    symbols: tuple[int, ...] = (0, 4, 7, 11)
    spacing: int = 6
    offsets: tuple[int, ...] = (0, 3, 0, 3)
    seed: int = 1234


@dataclass(frozen=True)
class ChannelConfig:
    n_sinusoids: int = 16
    max_taps: int = 32
    pdp_floor_db: float = -30.0


@dataclass(frozen=True)
class Region:
    """Range of channel conditions; each slot draws uniformly from it."""

    delay_spread_us: tuple[float, float]
    doppler_hz: tuple[float, float]


@dataclass(frozen=True)
class EvalConfig:
    snr_db: tuple[float, ...] = (0, 5, 10, 15, 20, 25, 30)
    n_slots: int = 1000
    batch_size: int = 100


@dataclass(frozen=True)
class Config:
    seed: int = 42
    ofdm: OFDMConfig = field(default_factory=OFDMConfig)
    pilots: PilotConfig = field(default_factory=PilotConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    regions: dict[str, Region] = field(default_factory=dict)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


def _tuples(d: dict[str, Any]) -> dict[str, Any]:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}


def config_from_dict(raw: dict[str, Any]) -> Config:
    pilots = PilotConfig(**_tuples(raw.get("pilots", {})))
    if len(pilots.offsets) != len(pilots.symbols):
        raise ValueError("pilots.offsets needs one entry per pilot symbol")
    return Config(
        seed=int(raw.get("seed", 42)),
        ofdm=OFDMConfig(**raw.get("ofdm", {})),
        pilots=pilots,
        channel=ChannelConfig(**raw.get("channel", {})),
        regions={name: Region(**_tuples(r)) for name, r in raw.get("regions", {}).items()},
        evaluation=EvalConfig(**_tuples(raw.get("evaluation", {}))),
    )


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return config_from_dict(yaml.safe_load(f))
