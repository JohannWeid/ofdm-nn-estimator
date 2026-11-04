"""Typed configuration for the neural estimator, loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Split:
    n_slots: int
    seed: int


@dataclass(frozen=True)
class DataConfig:
    region: str = "A"
    snr_db: tuple[float, float] = (0.0, 30.0)
    train: Split = field(default_factory=lambda: Split(32000, 100))
    val: Split = field(default_factory=lambda: Split(4000, 101))
    test_slots: int = 4000
    test_seeds: dict[str, int] = field(default_factory=lambda: {"A": 102, "B": 103})


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 16
    layers: int = 4
    kernel_size: int = 3
    use_snr: bool = True


@dataclass(frozen=True)
class TrainConfig:
    loss: str = "mse"
    batch_size: int = 128
    lr: float = 1e-3
    max_epochs: int = 100
    patience: int = 10


@dataclass(frozen=True)
class FailureMapConfig:
    delay_spread_us: tuple[float, ...] = (0.1, 0.5, 0.9, 1.3)
    doppler_hz: tuple[float, ...] = (0.0, 200.0, 400.0, 600.0)
    snr_db: tuple[float, ...] = (10.0, 20.0, 30.0)
    n_slots: int = 300
    seed: int = 104


@dataclass(frozen=True)
class NNConfig:
    name: str = "A"
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    failure_map: FailureMapConfig = field(default_factory=FailureMapConfig)


def nn_config_from_dict(raw: dict[str, Any]) -> NNConfig:
    data = dict(raw.get("data", {}))
    if "snr_db" in data:
        data["snr_db"] = tuple(data["snr_db"])
    for split in ("train", "val"):
        if split in data:
            data[split] = Split(**data[split])
    failure_map = {
        k: tuple(v) if isinstance(v, list) else v for k, v in raw.get("failure_map", {}).items()
    }
    return NNConfig(
        name=str(raw.get("name", "A")),
        seed=int(raw.get("seed", 0)),
        data=DataConfig(**data),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainConfig(**raw.get("training", {})),
        failure_map=FailureMapConfig(**failure_map),
    )


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw(path: Path) -> dict[str, Any]:
    """Read a YAML file; a `base` key names another file that this one only overrides."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    base = raw.pop("base", None)
    return _merge(_load_raw(path.parent / base), raw) if base else raw


def load_nn_config(path: str | Path) -> NNConfig:
    return nn_config_from_dict(_load_raw(Path(path)))
