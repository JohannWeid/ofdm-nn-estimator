"""Tests for training, saving and the NN config (Phase 3, step 3)."""

import csv
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nn.config import ModelConfig, TrainConfig, load_nn_config  # noqa: E402
from nn.dataset import make_dataset  # noqa: E402
from nn.model import build_model, load_model, save_model  # noqa: E402
from nn.train import (  # noqa: E402
    LOG_FIELDS,
    log_nmse_loss,
    make_loss,
    nmse_db,
    predict,
    train,
)
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


def test_experiment_config_only_overrides_what_it_names() -> None:
    base = load_nn_config(ROOT / "configs" / "nn.yaml")
    experiment = load_nn_config(ROOT / "configs" / "experiments" / "mse.yaml")
    assert experiment.name != base.name
    assert experiment.training.loss == "mse" and base.training.loss == "log_nmse"
    # Each config's name must point at the weights it produced.
    assert (ROOT / "models" / f"cnn_{base.name}.pt").exists()
    assert (ROOT / "models" / f"cnn_{experiment.name}.pt").exists()
    # Everything else must match, otherwise the comparison between the two would be unfair.
    assert experiment.data == base.data
    assert experiment.model == base.model
    assert experiment.seed == base.seed
    assert experiment.training.lr == base.training.lr
    assert experiment.training.max_epochs == base.training.max_epochs


def test_log_nmse_counts_every_slot_equally() -> None:
    target = torch.ones(2, 2, 14, 72)
    prediction = target.clone()
    prediction[0] += 0.1  # slot NMSE 1e-2 = -20 dB
    prediction[1] += 0.001  # slot NMSE 1e-6 = -60 dB
    assert log_nmse_loss(prediction, target).item() == pytest.approx(-40.0, abs=0.01)
    # An unweighted MSE is almost entirely the noisy slot.
    mse = torch.nn.functional.mse_loss(prediction, target).item()
    assert mse == pytest.approx(0.5 * 0.1**2, rel=1e-3)


def test_unknown_loss_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown loss"):
        make_loss("nmse")


def test_training_with_mse_improves_the_input(small_data) -> None:
    train_data, val_data = small_data
    cfg = TrainConfig(loss="mse", batch_size=32, lr=3e-3, max_epochs=6, patience=10)
    result = train(build_model(ModelConfig()), train_data, val_data, cfg, seed=0, verbose=False)
    x_val, y_val = (torch.from_numpy(a) for a in val_data)
    assert result.best_val_nmse_db < nmse_db(x_val[:, :2], y_val)
    assert result.best_val_loss < torch.nn.functional.mse_loss(x_val[:, :2], y_val).item()


def test_training_with_log_nmse_improves_the_input(small_data) -> None:
    train_data, val_data = small_data
    cfg = TrainConfig(loss="log_nmse", batch_size=32, lr=3e-3, max_epochs=6, patience=10)
    result = train(build_model(ModelConfig()), train_data, val_data, cfg, seed=0, verbose=False)
    x_val, y_val = (torch.from_numpy(a) for a in val_data)
    assert result.best_val_nmse_db < nmse_db(x_val[:, :2], y_val)
    assert result.best_val_loss < log_nmse_loss(x_val[:, :2], y_val).item()


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


def flat_weights(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.flatten() for p in model.parameters()]).detach()


def test_seed_fixes_the_start_weights() -> None:
    # Random state in between must not matter, as in two separate script runs.
    first = flat_weights(build_model(ModelConfig(), seed=0))
    torch.rand(7)
    assert torch.equal(first, flat_weights(build_model(ModelConfig(), seed=0)))
    assert not torch.equal(first, flat_weights(build_model(ModelConfig(), seed=1)))


def test_same_seed_gives_same_trained_weights(small_data) -> None:
    train_data, val_data = small_data
    cfg = TrainConfig(batch_size=64, lr=1e-3, max_epochs=2, patience=10)
    weights = []
    for _ in range(2):
        torch.rand(3)  # the global random state differs between the runs
        model = build_model(ModelConfig(), seed=0)
        train(model, train_data, val_data, cfg, seed=0, verbose=False)
        weights.append(flat_weights(model))
    torch.testing.assert_close(weights[0], weights[1])


def test_all_experiment_configs_load_and_name_their_own_model() -> None:
    base = load_nn_config(ROOT / "configs" / "nn.yaml")
    configs = [load_nn_config(p) for p in (ROOT / "configs" / "experiments").glob("*.yaml")]
    assert len(configs) >= 6
    names = [c.name for c in configs] + [base.name]
    assert len(set(names)) == len(names), "a shared name would overwrite another model"
    for c in configs:
        # The evaluation must stay comparable across all variants.
        assert c.data.val == base.data.val
        assert c.data.test_seeds == base.data.test_seeds
        assert c.data.train.seed == base.data.train.seed


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
