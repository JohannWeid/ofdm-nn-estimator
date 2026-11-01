"""Tests for the training data (Phase 3, step 1). Pure NumPy, no torch needed."""

import numpy as np
import pytest

from nn.dataset import SNR_SCALE_DB, make_dataset, to_input
from ofdm import LinkSimulator, evaluate
from ofdm.config import Config
from ofdm.estimators import InterpolationEstimator

S, K = 14, 72


def nmse_db(x: np.ndarray, y: np.ndarray) -> float:
    """NMSE between the first two channels of x (estimate) and y (truth) in dB."""
    err = np.sum((x[:, 0] - y[:, 0]) ** 2 + (x[:, 1] - y[:, 1]) ** 2)
    return float(10 * np.log10(err / np.sum(y[:, 0] ** 2 + y[:, 1] ** 2)))


def test_to_input_layout() -> None:
    rng = np.random.default_rng(0)
    h = rng.standard_normal((2, S, K)) + 1j * rng.standard_normal((2, S, K))
    x = to_input(h, np.array([0.0, 15.0]))
    assert x.shape == (2, 3, S, K)
    assert x.dtype == np.float32
    np.testing.assert_allclose(x[:, 0], h.real, rtol=1e-6)
    np.testing.assert_allclose(x[:, 1], h.imag, rtol=1e-6)
    np.testing.assert_allclose(x[0, 2], 0.0)
    np.testing.assert_allclose(x[1, 2], 15.0 / SNR_SCALE_DB, rtol=1e-6)


def test_shapes_and_dtypes_with_partial_last_batch(sim: LinkSimulator, cfg: Config) -> None:
    x, y = make_dataset(sim, cfg.regions["A"], 300, (0.0, 30.0), seed=1, batch_size=128)
    assert x.shape == (300, 3, S, K)
    assert y.shape == (300, 2, S, K)
    assert x.dtype == np.float32 and y.dtype == np.float32
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))


def test_same_seed_gives_identical_data(sim: LinkSimulator, cfg: Config) -> None:
    a = make_dataset(sim, cfg.regions["A"], 50, (0.0, 30.0), seed=7)
    b = make_dataset(sim, cfg.regions["A"], 50, (0.0, 30.0), seed=7)
    c = make_dataset(sim, cfg.regions["A"], 50, (0.0, 30.0), seed=8)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_snr_plane_is_constant_per_slot_and_in_range(sim: LinkSimulator, cfg: Config) -> None:
    x, _ = make_dataset(sim, cfg.regions["A"], 200, (5.0, 25.0), seed=2)
    plane = x[:, 2]
    np.testing.assert_array_equal(plane, np.broadcast_to(plane[:, :1, :1], plane.shape))
    snr = plane[:, 0, 0] * SNR_SCALE_DB
    assert snr.min() >= 5.0 - 1e-4 and snr.max() <= 25.0 + 1e-4
    assert snr.max() - snr.min() > 10.0, "SNR should be drawn per slot, not fixed"


def test_targets_have_unit_average_power(sim: LinkSimulator, cfg: Config) -> None:
    _, y = make_dataset(sim, cfg.regions["A"], 600, (10.0, 10.0), seed=3)
    assert np.mean(y[:, 0] ** 2 + y[:, 1] ** 2) == pytest.approx(1.0, abs=0.15)


def test_input_error_matches_ls_linear_baseline(sim: LinkSimulator, cfg: Config) -> None:
    """At a fixed SNR the input must have the same NMSE as the LS + linear baseline."""
    x, y = make_dataset(sim, cfg.regions["A"], 400, (10.0, 10.0), seed=4)
    rows = evaluate(
        sim,
        {"ls_linear": InterpolationEstimator(sim.pattern, "linear")},
        cfg.regions["A"],
        [10.0],
        n_slots=400,
        batch_size=200,
        rng=np.random.default_rng(5),
    )
    assert nmse_db(x, y) == pytest.approx(rows[0]["nmse_db"], abs=1.0)
