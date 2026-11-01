"""Training data for the neural channel estimator.

One sample is one slot:
    input  (3, S, K) float32: Re and Im of the LS + linear estimate, plus an SNR plane
    target (2, S, K) float32: Re and Im of the true channel h_true
"""

from __future__ import annotations

import numpy as np

from ofdm import LinkSimulator, sample_profiles
from ofdm.config import Region
from ofdm.estimators import InterpolationEstimator

SNR_SCALE_DB = 30.0  # the SNR plane holds snr_db / SNR_SCALE_DB


def to_input(h_interp: np.ndarray, snr_db: np.ndarray) -> np.ndarray:
    """Build the network input from an interpolated channel estimate.

    Training and inference both call this function, so they see identical preprocessing.

    Args:
        h_interp: (B, S, K) complex, e.g. InterpolationEstimator(...).interpolate(rx_grid).
        snr_db: (B,) SNR of each slot in dB.

    Returns:
        (B, 3, S, K) float32 array. Channel 0 holds the real part, channel 1 the imaginary
        part, and channel 2 holds snr_db / SNR_SCALE_DB, repeated over the whole grid.
    """
    B, S, K = h_interp.shape
    snr = np.asarray(snr_db, dtype=float) / SNR_SCALE_DB
    snr_plane = np.broadcast_to(snr[:, None, None], (B, S, K))
    return np.stack([h_interp.real, h_interp.imag, snr_plane], axis=1).astype(np.float32)


def make_dataset(
    sim: LinkSimulator,
    region: Region,
    n_slots: int,
    snr_range_db: tuple[float, float],
    seed: int,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate n_slots slots and return (inputs, targets) as float32 arrays.

    inputs has shape (n_slots, 3, S, K) and comes from to_input. targets has shape
    (n_slots, 2, S, K) and holds Re and Im of h_true. Every slot draws its delay spread
    and Doppler from `region` and its SNR uniformly from `snr_range_db`.
    The same seed must always give identical arrays.
    """
    rng = np.random.default_rng(seed)  # the only source of randomness -> reproducible
    interp = InterpolationEstimator(sim.pattern, "linear")
    S, K = sim.pattern.shape
    inputs = np.empty((n_slots, 3, S, K), dtype=np.float32)
    targets = np.empty((n_slots, 2, S, K), dtype=np.float32)

    for start in range(0, n_slots, batch_size):
        b = min(batch_size, n_slots - start)
        snr_db = rng.uniform(*snr_range_db, size=b)
        slots = sim.simulate(b, snr_db, rng, sample_profiles(region, b, rng))
        out = slice(start, start + b)
        inputs[out] = to_input(interp.interpolate(slots.rx_grid), slots.snr_db)
        targets[out, 0] = slots.h_true.real
        targets[out, 1] = slots.h_true.imag
    return inputs, targets
