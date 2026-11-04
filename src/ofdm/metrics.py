"""Performance metrics and closed-form references."""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

from .modulation import bits_per_symbol


def to_db(x: np.ndarray | float) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 10 * np.log10(x)


def nmse(h_hat: np.ndarray, h_true: np.ndarray) -> float:
    """sum |H_hat - H|^2 / sum |H|^2 over all resource elements and slots."""
    return float(np.sum(np.abs(h_hat - h_true) ** 2) / np.sum(np.abs(h_true) ** 2))


def tracked_variation(h_hat: np.ndarray, h_true: np.ndarray, axis: int, lag: int) -> float:
    """Share of the true channel change over `lag` cells that the estimate reproduces.

    Regresses the change of the estimate onto the true change: 1 means the estimate follows
    the channel, below 1 means it smooths the variation away. Noise is uncorrelated with the
    true change, so it does not bias this measure. Returns NaN for a channel without change.
    """

    def change(h: np.ndarray) -> np.ndarray:
        n = h.shape[axis]
        return np.take(h, range(lag, n), axis) - np.take(h, range(n - lag), axis)

    d_true, d_hat = change(h_true), change(h_hat)
    energy = np.sum(np.abs(d_true) ** 2)
    if energy == 0:
        return float("nan")
    return float(np.real(np.sum(np.conj(d_true) * d_hat)) / energy)


def bit_error_rate(bits_hat: np.ndarray, bits: np.ndarray) -> float:
    return float(np.mean(bits_hat != bits))


def q_function(x: np.ndarray) -> np.ndarray:
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2))


def ber_awgn(snr_db: np.ndarray | float, modulation: str) -> np.ndarray:
    """Exact BER of Gray-coded QPSK / 16-QAM in AWGN; SNR = Es/N0 per resource element."""
    snr = 10 ** (np.asarray(snr_db, dtype=float) / 10)
    q = bits_per_symbol(modulation)
    if q == 2:
        return q_function(np.sqrt(snr))
    if q == 4:
        x = np.sqrt(snr / 5)  # half the distance between levels over the noise std per axis
        return (3 * q_function(x) + 2 * q_function(3 * x) - q_function(5 * x)) / 4
    raise ValueError(f"no closed form for {modulation!r}")


def ebn0_db(snr_db: np.ndarray | float, modulation: str) -> np.ndarray:
    """Eb/N0 = Es/N0 / bits per symbol (pilot and CP overhead not included)."""
    return np.asarray(snr_db, dtype=float) - 10 * np.log10(bits_per_symbol(modulation))
