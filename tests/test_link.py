import numpy as np
import pytest

from ofdm import LinkSimulator, PerfectCSI, evaluate, fixed_profiles
from ofdm.config import config_from_dict
from ofdm.estimators import InterpolationEstimator
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
