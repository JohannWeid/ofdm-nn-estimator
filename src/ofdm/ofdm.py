"""OFDM modulation (IFFT + cyclic prefix) and demodulation (CP removal + FFT).

The FFTs use norm="ortho", so white noise with variance sigma^2 per time sample
has variance sigma^2 on every subcarrier. The SNR is therefore defined per
resource element: SNR = E|H X|^2 / sigma^2 = 1 / sigma^2.
"""

from __future__ import annotations

import numpy as np

from .config import OFDMConfig


def subcarrier_bins(cfg: OFDMConfig) -> np.ndarray:
    """FFT bin of each active subcarrier; the band is centered around DC."""
    k = np.arange(cfg.n_subcarriers) - cfg.n_subcarriers // 2
    return k % cfg.n_fft


def modulate(grid: np.ndarray, cfg: OFDMConfig) -> np.ndarray:
    """Resource grid (B, S, K) -> time-domain slot (B, S * (n_fft + cp))."""
    B, S, K = grid.shape
    spectrum = np.zeros((B, S, cfg.n_fft), dtype=np.complex128)
    spectrum[..., subcarrier_bins(cfg)] = grid
    symbols = np.fft.ifft(spectrum, axis=-1, norm="ortho")
    with_cp = np.concatenate([symbols[..., cfg.n_fft - cfg.cp_length :], symbols], axis=-1)
    return with_cp.reshape(B, S * cfg.symbol_length)


def demodulate(signal: np.ndarray, cfg: OFDMConfig) -> np.ndarray:
    """Time-domain slot (B, S * (n_fft + cp)) -> resource grid (B, S, K)."""
    B = signal.shape[0]
    symbols = signal.reshape(B, cfg.n_symbols, cfg.symbol_length)[..., cfg.cp_length :]
    spectrum = np.fft.fft(symbols, axis=-1, norm="ortho")
    return spectrum[..., subcarrier_bins(cfg)]
