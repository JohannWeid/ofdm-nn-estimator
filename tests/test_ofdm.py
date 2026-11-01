import numpy as np

from ofdm import ofdm
from ofdm.config import Config


def test_fft_ifft_roundtrip(cfg: Config, rng: np.random.Generator) -> None:
    shape = (4, cfg.ofdm.n_symbols, cfg.ofdm.n_subcarriers)
    grid = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    signal = ofdm.modulate(grid, cfg.ofdm)
    assert signal.shape == (4, cfg.ofdm.slot_length)
    np.testing.assert_allclose(ofdm.demodulate(signal, cfg.ofdm), grid, atol=1e-12)


def test_cyclic_prefix_is_copy_of_symbol_end(cfg: Config, rng: np.random.Generator) -> None:
    shape = (1, cfg.ofdm.n_symbols, cfg.ofdm.n_subcarriers)
    grid = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    symbols = ofdm.modulate(grid, cfg.ofdm).reshape(cfg.ofdm.n_symbols, cfg.ofdm.symbol_length)
    cp = cfg.ofdm.cp_length
    np.testing.assert_allclose(symbols[:, :cp], symbols[:, -cp:])


def test_ortho_fft_keeps_noise_variance_per_subcarrier(cfg: Config) -> None:
    rng = np.random.default_rng(1)
    n = (200, cfg.ofdm.slot_length)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    per_re = ofdm.demodulate(noise, cfg.ofdm)
    assert abs(np.mean(np.abs(per_re) ** 2) - 1.0) < 0.02
