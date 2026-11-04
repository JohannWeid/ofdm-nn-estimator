import numpy as np
import pytest

from ofdm import LinkSimulator, PerfectCSI, evaluate, fixed_profiles
from ofdm.config import config_from_dict
from ofdm.estimators import InterpolationEstimator
from ofdm.link import evaluate_grid
from ofdm.metrics import ber_awgn


def test_ber_is_zero_without_noise_awgn_channel(sim: LinkSimulator) -> None:
    rng = np.random.default_rng(4)
    slots = sim.simulate(8, np.inf, rng)
    h_hat = InterpolationEstimator(sim.pattern, "linear")(slots)
    assert sim.bit_errors(slots, h_hat) == 0


def test_ber_is_zero_without_noise_static_fading(sim: LinkSimulator) -> None:
    """Delay spread inside the CP and no Doppler: no ISI/ICI, so ZF with true H is exact."""
    rng = np.random.default_rng(5)
    slots = sim.simulate(8, np.inf, rng, fixed_profiles(0.5e-6, 0.0, 8))
    assert sim.bit_errors(slots, PerfectCSI()(slots)) == 0


@pytest.mark.parametrize("modulation,snr_db", [("qpsk", 4.0), ("qpsk", 8.0), ("16qam", 12.0)])
def test_awgn_ber_matches_theory(modulation: str, snr_db: float) -> None:
    sim = LinkSimulator(config_from_dict({"ofdm": {"modulation": modulation}}))
    rows = evaluate(
        sim, {"perfect": PerfectCSI()}, None, [snr_db], n_slots=200, batch_size=100,
        rng=np.random.default_rng(6),
    )
    theory = float(ber_awgn(snr_db, modulation))
    assert rows[0]["ber"] == pytest.approx(theory, rel=0.1)


def test_snr_per_slot_array(sim: LinkSimulator) -> None:
    rng = np.random.default_rng(7)
    snr = np.array([0.0, 10.0, 20.0])
    slots = sim.simulate(3, snr, rng, fixed_profiles(0.2e-6, 50.0, 3))
    np.testing.assert_allclose(slots.snr_db, snr)


def test_evaluate_grid_covers_all_cells_and_sees_channel_conditions(sim: LinkSimulator) -> None:
    rows = evaluate_grid(
        sim, {"ls_linear": InterpolationEstimator(sim.pattern, "linear")},
        delay_spreads_us=[0.1, 1.3], dopplers_hz=[0.0, 600.0], snr_db=[30.0],
        n_slots=30, batch_size=30, rng=np.random.default_rng(8),
    )
    assert len(rows) == 4
    nmse = {(r["delay_spread_us"], r["doppler_hz"]): r["nmse_db"] for r in rows}
    # Interpolation gets harder with a larger delay spread and a higher Doppler.
    assert nmse[(1.3, 0.0)] > nmse[(0.1, 0.0)] + 3.0
    assert nmse[(0.1, 600.0)] > nmse[(0.1, 0.0)] + 3.0
