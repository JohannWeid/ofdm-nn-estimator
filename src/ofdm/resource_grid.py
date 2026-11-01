"""Resource grid layout: which resource elements (REs) carry pilots and which carry data.

Grids have shape (n_symbols, n_subcarriers), i.e. time first, frequency second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import OFDMConfig, PilotConfig


@dataclass(frozen=True)
class PilotPattern:
    symbols: np.ndarray  # (P_t,) OFDM symbol indices that carry pilots
    subcarriers: tuple[np.ndarray, ...]  # per pilot symbol: sorted pilot subcarrier indices
    mask: np.ndarray  # (S, K) bool, True at pilot REs
    values: np.ndarray  # (S, K) complex, known pilot symbols (0 at data REs)

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape

    @property
    def n_pilots(self) -> int:
        return int(self.mask.sum())

    @property
    def n_data(self) -> int:
        return int((~self.mask).sum())

    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        """(symbol, subcarrier) indices of all pilots in row-major order."""
        return np.nonzero(self.mask)


def make_pilot_pattern(ofdm: OFDMConfig, pilots: PilotConfig) -> PilotPattern:
    S, K = ofdm.n_symbols, ofdm.n_subcarriers
    symbols = np.asarray(pilots.symbols, dtype=np.int64)
    if symbols.min() < 0 or symbols.max() >= S or np.any(np.diff(symbols) <= 0):
        raise ValueError("pilot symbols must be strictly increasing and inside the slot")

    mask = np.zeros((S, K), dtype=bool)
    subcarriers = []
    for s, off in zip(symbols, pilots.offsets, strict=True):
        sc = np.arange(off % pilots.spacing, K, pilots.spacing)
        mask[s, sc] = True
        subcarriers.append(sc)

    # Fixed unit-modulus QPSK pilot sequence, independent of the simulation seed.
    rng = np.random.default_rng(pilots.seed)
    phases = rng.integers(0, 4, size=(S, K))
    values = np.where(mask, np.exp(1j * (np.pi / 4 + np.pi / 2 * phases)), 0.0)
    return PilotPattern(symbols, tuple(subcarriers), mask, values)
