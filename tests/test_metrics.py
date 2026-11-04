import numpy as np
import pytest

from ofdm.metrics import snr_at_ber, tracked_variation


def smooth_channel(rng: np.random.Generator) -> np.ndarray:
    """Batch of random grids (B, S, K) whose values change from cell to cell."""
    shape = (50, 14, 72)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_perfect_estimate_tracks_everything() -> None:
    h = smooth_channel(np.random.default_rng(0))
    assert tracked_variation(h, h, axis=1, lag=3) == pytest.approx(1.0)


def test_scaled_estimate_tracks_its_share() -> None:
    h = smooth_channel(np.random.default_rng(1))
    assert tracked_variation(0.6 * h, h, axis=2, lag=6) == pytest.approx(0.6)


def test_noise_does_not_bias_the_measure() -> None:
    rng = np.random.default_rng(2)
    h = smooth_channel(rng)
    noise = rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape)
    assert tracked_variation(h + noise, h, axis=1, lag=1) == pytest.approx(1.0, abs=0.02)


def test_constant_estimate_tracks_nothing_and_static_channel_gives_nan() -> None:
    h = smooth_channel(np.random.default_rng(3))
    assert tracked_variation(np.ones_like(h), h, axis=1, lag=1) == pytest.approx(0.0)
    assert np.isnan(tracked_variation(h, np.ones_like(h), axis=1, lag=1))


def test_snr_at_ber_interpolates_in_log_ber() -> None:
    assert snr_at_ber([0.0, 10.0, 20.0], [1e-1, 1e-3, 1e-4], 1e-2) == pytest.approx(5.0)
    assert snr_at_ber([0.0, 10.0, 20.0], [1e-1, 1e-3, 1e-4], 3e-4) == pytest.approx(15.23, abs=0.01)


def test_snr_at_ber_is_nan_without_a_usable_crossing() -> None:
    assert np.isnan(snr_at_ber([0.0, 10.0], [1e-1, 5e-2], 1e-2))  # stays above the target
    assert np.isnan(snr_at_ber([0.0, 10.0], [1e-3, 1e-4], 1e-2))  # starts below the target
    assert np.isnan(snr_at_ber([0.0, 10.0], [1e-1, 0.0], 1e-2))  # log(0) cannot be interpolated
