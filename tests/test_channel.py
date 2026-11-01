import numpy as np
import pytest

from ofdm import fixed_profiles, sample_profiles
from ofdm.channel import exponential_pdp, frequency_response, generate_taps
from ofdm.config import Config


@pytest.mark.parametrize("tau_us", [0.0, 0.1, 0.5, 1.3])
def test_pdp_is_normalized_and_decaying(tau_us: float) -> None:
    p = exponential_pdp(tau_us * 1e-6, 1 / 1.92e6, max_taps=32, floor_db=-30.0)
    assert p.sum() == pytest.approx(1.0)
    assert np.all(np.diff(p) <= 0)


def test_average_channel_power_is_one(cfg: Config) -> None:
    rng = np.random.default_rng(2)
    power = []
    for _ in range(4):  # 800 slots in batches to bound memory
        profiles = fixed_profiles(1.0e-6, 50.0, 200)
        h = frequency_response(generate_taps(profiles, cfg.ofdm, cfg.channel, rng), cfg.ofdm)
        power.append(np.mean(np.abs(h) ** 2))
    assert np.mean(power) == pytest.approx(1.0, abs=0.06)


def test_sampled_profiles_stay_inside_region(cfg: Config, rng: np.random.Generator) -> None:
    region = cfg.regions["B"]
    p = sample_profiles(region, 500, rng)
    assert np.all(p.delay_spread_s >= region.delay_spread_us[0] * 1e-6)
    assert np.all(p.delay_spread_s <= region.delay_spread_us[1] * 1e-6)
    assert np.all((p.doppler_hz >= region.doppler_hz[0]) & (p.doppler_hz <= region.doppler_hz[1]))


def test_static_channel_does_not_change_over_slot(cfg: Config, rng: np.random.Generator) -> None:
    taps = generate_taps(fixed_profiles(0.5e-6, 0.0, 3), cfg.ofdm, cfg.channel, rng).taps
    np.testing.assert_allclose(taps, taps[..., :1].repeat(taps.shape[-1], axis=-1))


def test_doppler_decorrelates_channel_over_time(cfg: Config) -> None:
    """Correlation between the first and last OFDM symbol drops with Doppler."""

    def corr(doppler_hz: float) -> float:
        rng = np.random.default_rng(3)
        profiles = fixed_profiles(0.3e-6, doppler_hz, 400)
        h = frequency_response(generate_taps(profiles, cfg.ofdm, cfg.channel, rng), cfg.ofdm)
        first, last = h[:, 0], h[:, -1]
        return float(np.abs(np.mean(first * last.conj())) / np.mean(np.abs(first) ** 2))

    assert corr(10.0) > 0.95
    assert corr(600.0) < corr(10.0) - 0.2
