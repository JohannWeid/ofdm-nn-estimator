"""Tests for the CNN as a channel estimator (Phase 3, step 4)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nn.config import ModelConfig  # noqa: E402
from nn.estimator import NeuralEstimator  # noqa: E402
from nn.model import build_model, load_model  # noqa: E402
from ofdm import LinkSimulator, evaluate, sample_profiles  # noqa: E402
from ofdm.config import Config  # noqa: E402
from ofdm.estimators import InterpolationEstimator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_untrained_model_reproduces_ls_linear(sim: LinkSimulator, cfg: Config) -> None:
    rng = np.random.default_rng(20)
    slots = sim.simulate(4, 10.0, rng, sample_profiles(cfg.regions["A"], 4, rng))
    h_cnn = NeuralEstimator(build_model(ModelConfig()), sim.pattern)(slots)
    h_ls = InterpolationEstimator(sim.pattern, "linear")(slots)
    assert h_cnn.shape == slots.h_true.shape
    assert np.iscomplexobj(h_cnn)
    np.testing.assert_allclose(h_cnn, h_ls, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("model_file", ["cnn_A_log_nmse.pt", "cnn_A.pt"])
def test_committed_models_beat_ls_linear_in_region_a(
    sim: LinkSimulator, cfg: Config, model_file: str
) -> None:
    """End-to-end check of the stored weights through the normal evaluation path."""
    estimators = {
        "ls_linear": InterpolationEstimator(sim.pattern, "linear"),
        "cnn": NeuralEstimator(load_model(ROOT / "models" / model_file), sim.pattern),
    }
    rows = evaluate(
        sim, estimators, cfg.regions["A"], [10.0], n_slots=100, batch_size=100,
        rng=np.random.default_rng(21),
    )
    nmse = {r["estimator"]: r["nmse_db"] for r in rows}
    ber = {r["estimator"]: r["ber"] for r in rows}
    assert nmse["cnn"] < nmse["ls_linear"] - 2.0
    assert ber["cnn"] < ber["ls_linear"]
