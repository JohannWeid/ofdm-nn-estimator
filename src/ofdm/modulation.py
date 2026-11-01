"""Gray-coded square QAM (QPSK, 16-QAM) with unit average symbol energy.

Bits of one symbol are split alternately onto the I and Q axis (b0 -> I, b1 -> Q, ...).
Each axis is a Gray-coded PAM, so hard decisions can be made per axis.
"""

from __future__ import annotations

import numpy as np

BITS_PER_SYMBOL = {"qpsk": 2, "16qam": 4}


def bits_per_symbol(modulation: str) -> int:
    try:
        return BITS_PER_SYMBOL[modulation.lower()]
    except KeyError:
        raise ValueError(f"unknown modulation {modulation!r}") from None


def _axis_tables(modulation: str) -> tuple[int, np.ndarray, np.ndarray, float]:
    """Return (bits per axis, gray code per level, level per gray code, scale)."""
    q = bits_per_symbol(modulation)
    bpa = q // 2
    n_levels = 2**bpa
    idx = np.arange(n_levels)
    gray = idx ^ (idx >> 1)  # level index -> gray label
    level_of_gray = np.argsort(gray)  # gray label -> level index
    scale = np.sqrt(2.0 * (2**q - 1) / 3.0)  # average energy of the unnormalized grid
    return bpa, gray, level_of_gray, scale


def _bits_to_int(bits: np.ndarray) -> np.ndarray:
    weights = 1 << np.arange(bits.shape[-1] - 1, -1, -1)
    return (bits.astype(np.int64) * weights).sum(axis=-1)


def _int_to_bits(values: np.ndarray, n_bits: int) -> np.ndarray:
    shifts = np.arange(n_bits - 1, -1, -1)
    return ((values[..., None] >> shifts) & 1).astype(np.uint8)


def modulate(bits: np.ndarray, modulation: str) -> np.ndarray:
    """Map bits (..., n_symbols * q) to complex symbols (..., n_symbols)."""
    q = bits_per_symbol(modulation)
    bpa, _, level_of_gray, scale = _axis_tables(modulation)
    if bits.shape[-1] % q:
        raise ValueError(f"number of bits must be a multiple of {q}")
    b = bits.reshape(*bits.shape[:-1], -1, q)
    n_levels = 2**bpa
    i_level = level_of_gray[_bits_to_int(b[..., 0::2])]
    q_level = level_of_gray[_bits_to_int(b[..., 1::2])]
    # level index 0..n-1 -> amplitude -(n-1), ..., -1, 1, ..., (n-1)
    i_amp = 2 * i_level - (n_levels - 1)
    q_amp = 2 * q_level - (n_levels - 1)
    return (i_amp + 1j * q_amp) / scale


def demodulate(symbols: np.ndarray, modulation: str) -> np.ndarray:
    """Hard-decision demapping of complex symbols (..., n) to bits (..., n * q)."""
    q = bits_per_symbol(modulation)
    bpa, gray, _, scale = _axis_tables(modulation)
    n_levels = 2**bpa

    def axis_bits(x: np.ndarray) -> np.ndarray:
        level = np.clip(np.round((x * scale + (n_levels - 1)) / 2), 0, n_levels - 1)
        return _int_to_bits(gray[level.astype(np.int64)], bpa)

    out = np.empty((*symbols.shape, q), dtype=np.uint8)
    out[..., 0::2] = axis_bits(symbols.real)
    out[..., 1::2] = axis_bits(symbols.imag)
    return out.reshape(*symbols.shape[:-1], -1)


def constellation(modulation: str) -> np.ndarray:
    """All constellation points, e.g. for plotting."""
    q = bits_per_symbol(modulation)
    labels = _int_to_bits(np.arange(2**q), q)
    return modulate(labels, modulation).ravel()
