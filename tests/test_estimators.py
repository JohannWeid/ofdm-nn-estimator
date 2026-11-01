import numpy as np

from ofdm import LinkSimulator, baseline_estimators, evaluate, fixed_profiles
from ofdm.estimators import (
    linear_interpolation_matrix,
    ls_at_pilots,
    spline_interpolation_matrix,
)


def test_linear_interpolation_is_exact_for_lines() -> None:
    x_known = np.array([1.0, 4.0, 9.0])
    x_new = np.arange(-2.0, 12.0)
    W = linear_interpolation_matrix(x_known, x_new)
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    np.testing.assert_allclose(W @ (3 * x_known - 2), 3 * x_new - 2)


def test_spline_interpolation_hits_knots_and_is_exact_for_lines() -> None:
    x_known = np.arange(3.0, 72.0, 6.0)
    x_new = np.arange(72.0)
    W = spline_interpolation_matrix(x_known, x_new)
    np.testing.assert_allclose(W[x_known.astype(int)], np.eye(x_known.size), atol=1e-12)
    np.testing.assert_allclose(W @ (0.5 * x_known - 4), 0.5 * x_new - 4, atol=1e-12)


def test_spline_extrapolates_linearly_outside_pilots() -> None:
    x_known = np.array([0.0, 4.0, 7.0, 11.0])
    x_new = np.array([12.0, 13.0])
    np.testing.assert_allclose(
        spline_interpolation_matrix(x_known, x_new), linear_interpolation_matrix(x_known, x_new)
    )


def test_ls_is_exact_at_pilots_without_noise(sim: LinkSimulator) -> None:
    rng = np.random.default_rng(8)
    slots = sim.simulate(4, np.inf, rng, fixed_profiles(0.5e-6, 0.0, 4))
    np.testing.assert_allclose(
        ls_at_pilots(slots.rx_grid, sim.pattern), slots.h_true[:, sim.pattern.mask], atol=1e-10
    )


def test_lmmse_beats_ls_at_low_snr(sim: LinkSimulator) -> None:
    est = baseline_estimators(sim.pattern, sim.ofdm)
    rows = evaluate(
        sim, est, sim.cfg.regions["A"], [0.0, 10.0], n_slots=60, batch_size=30,
        rng=np.random.default_rng(9),
    )
    nmse = {(r["snr_db"], r["estimator"]): r["nmse_db"] for r in rows}
    for snr in (0.0, 10.0):
        assert nmse[(snr, "lmmse")] < nmse[(snr, "ls_linear")] - 3.0
        assert nmse[(snr, "lmmse")] < nmse[(snr, "ls_spline")]


def test_nmse_decreases_with_snr(sim: LinkSimulator) -> None:
    est = baseline_estimators(sim.pattern, sim.ofdm)
    rows = evaluate(
        sim, {"ls_linear": est["ls_linear"]}, sim.cfg.regions["A"], [0.0, 10.0, 20.0],
        n_slots=40, batch_size=40, rng=np.random.default_rng(10),
    )
    nmse = [r["nmse_db"] for r in sorted(rows, key=lambda r: r["snr_db"])]
    assert np.all(np.isfinite(nmse))
    assert nmse[0] > nmse[1] > nmse[2]
