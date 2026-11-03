"""Tests for training, saving and the NN config (Phase 3, step 3)."""

import csv
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nn.config import ModelConfig, TrainConfig, load_nn_config  # noqa: E402
from nn.dataset import make_dataset  # noqa: E402
from nn.model import build_model, load_model, save_model  # noqa: E402
from nn.train import LOG_FIELDS, nmse_db, predict, train  # noqa: E402
from ofdm import LinkSimulator  # noqa: E402
from ofdm.config import Config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def small_data(sim: LinkSimulator, cfg: Config) -> tuple[tuple, tuple]:
    region = cfg.regions["A"]
    return (
        make_dataset(sim, region, 256, (0.0, 20.0), seed=10),
        make_dataset(sim, region, 128, (0.0, 20.0), seed=11),
    )


def test_nn_config_file_loads() -> None:
    nn_cfg = load_nn_config(ROOT / "configs" / "nn.yaml")
    assert nn_cfg.data.region == "A"
    assert nn_cfg.data.train.seed != nn_cfg.data.val.seed
    assert nn_cfg.model == ModelConfig()


def test_short_training_beats_the_input_and_writes_a_log(small_data, tmp_path: Path) -> None:
    train_data, val_data = small_data
    model = build_model(ModelConfig())
    cfg = TrainConfig(batch_size=32, lr=3e-3, max_epochs=6, patience=10)
    result = train(model, train_data, val_data, cfg, seed=0, log_path=tmp_path / "log.csv",
                   verbose=False)

    x_val, y_val = (torch.from_numpy(a) for a in val_data)
    assert result.best_val_nmse_db < nmse_db(x_val[:, :2], y_val)
    # the model is left at its best epoch
    assert nmse_db(predict(model, x_val), y_val) == pytest.approx(result.best_val_nmse_db)
    with open(tmp_path / "log.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == LOG_FIELDS
    assert len(rows) == len(result.history) == 6


def test_early_stopping(small_data) -> None:
    """With lr = 0 the val loss never improves, so training stops after `patience` epochs."""
    train_data, val_data = small_data
    cfg = TrainConfig(batch_size=64, lr=0.0, max_epochs=50, patience=2)
    result = train(build_model(ModelConfig()), train_data, val_data, cfg, seed=0, verbose=False)
    assert result.best_epoch == 1
    assert len(result.history) == 3


def test_same_seed_gives_same_weights(small_data) -> None:
    train_data, val_data = small_data
    cfg = TrainConfig(batch_size=64, lr=1e-3, max_epochs=2, patience=10)
    weights = []
    for _ in range(2):
        torch.manual_seed(123)  # identical init
        model = build_model(ModelConfig())
        train(model, train_data, val_data, cfg, seed=0, verbose=False)
        weights.append(torch.cat([p.flatten() for p in model.parameters()]))
    torch.testing.assert_close(weights[0], weights[1])


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    cfg = ModelConfig(channels=8, layers=3)
    model = build_model(cfg)
    torch.nn.init.normal_(model.body[-1].weight)
    save_model(model, cfg, tmp_path / "m.pt")
    loaded = load_model(tmp_path / "m.pt")
    x = torch.randn(2, 3, 14, 72)
    torch.testing.assert_close(loaded(x), model.eval()(x))
    assert np.isclose(sum(p.numel() for p in loaded.parameters()),
                      sum(p.numel() for p in model.parameters()))
