from pathlib import Path

import numpy as np
import pytest

from ofdm import Config, LinkSimulator, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def cfg() -> Config:
    return load_config(CONFIG_PATH)


@pytest.fixture
def sim(cfg: Config) -> LinkSimulator:
    return LinkSimulator(cfg)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
