"""Classical channel estimators.

Each estimator is a callable that maps a SlotBatch to an estimate H_hat of shape (B, S, K).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy.interpolate import CubicSpline

from .channel import frequency_correlation, time_correlation
from .config import OFDMConfig
from .resource_grid import PilotPattern

if TYPE_CHECKING:
    from .link import SlotBatch


class Estimator(Protocol):
    def __call__(self, slots: SlotBatch) -> np.ndarray: ...


def linear_interpolation_matrix(x_known: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Matrix W (len(x_new), len(x_known)) with y_new = W @ y_known.

    Piecewise linear; outside the known points the first/last segment is extrapolated.
    """
    x_known = np.asarray(x_known, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    n = x_known.size
    W = np.zeros((x_new.size, n))
    if n == 1:
        W[:, 0] = 1.0
        return W
    j = np.clip(np.searchsorted(x_known, x_new, side="right"), 1, n - 1)
    x0, x1 = x_known[j - 1], x_known[j]
    w1 = (x_new - x0) / (x1 - x0)
    rows = np.arange(x_new.size)
    W[rows, j - 1] = 1.0 - w1
    W[rows, j] = w1
    return W


def spline_interpolation_matrix(x_known: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Natural cubic spline as a matrix (the spline is linear in the known values).

    Outside the known points it extrapolates linearly like linear_interpolation_matrix.
    Extrapolating the cubic itself would amplify pilot noise at the grid edges by up to
    ~18 dB, which would make this baseline unfairly weak.
    """
    x_known = np.asarray(x_known, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    if x_known.size < 3:
        return linear_interpolation_matrix(x_known, x_new)
    W = linear_interpolation_matrix(x_known, x_new)
    inside = (x_new >= x_known[0]) & (x_new <= x_known[-1])
    basis = CubicSpline(x_known, np.eye(x_known.size), axis=0, bc_type="natural")
    W[inside] = basis(x_new[inside])
    return W


def ls_at_pilots(rx_grid: np.ndarray, pattern: PilotPattern) -> np.ndarray:
    """Least-squares estimate Y_p / X_p at all pilots, row-major order: (B, P)."""
    return rx_grid[:, pattern.mask] / pattern.values[pattern.mask]


class InterpolationEstimator:
    """LS at the pilots, then separable interpolation: first over frequency, then time."""

    def __init__(self, pattern: PilotPattern, kind: str = "linear") -> None:
        make = {"linear": linear_interpolation_matrix, "spline": spline_interpolation_matrix}[kind]
        S, K = pattern.shape
        self.pattern = pattern
        self.freq = [make(sc, np.arange(K)) for sc in pattern.subcarriers]  # each (K, n_i)
        self.time = make(pattern.symbols, np.arange(S))  # (S, P_t)

    def __call__(self, slots: SlotBatch) -> np.ndarray:
        return self.interpolate(slots.rx_grid)

    def interpolate(self, rx_grid: np.ndarray) -> np.ndarray:
        p = self.pattern
        rows = []
        for s, sc, W in zip(p.symbols, p.subcarriers, self.freq, strict=True):
            h_ls = rx_grid[:, s, sc] / p.values[s, sc]  # (B, n_i)
            rows.append(h_ls @ W.T)  # (B, K)
        h_freq = np.stack(rows, axis=1)  # (B, P_t, K)
        return np.einsum("st,btk->bsk", self.time, h_freq)


class LMMSEEstimator:
    """2D linear MMSE (Wiener) estimator with the true per-slot channel statistics.

    H_hat = R_hp (R_pp + sigma^2 I)^-1 h_ls_p, with R(dk, dm) = r_f(dk) * r_t(dm) for a
    WSSUS channel. Knowing the exact PDP, Doppler and noise variance is a genie
    assumption, so this estimator is the upper bound among the baselines.
    """

    def __init__(self, pattern: PilotPattern, ofdm: OFDMConfig, chunk: int = 32) -> None:
        S, K = pattern.shape
        self.pattern = pattern
        self.ofdm = ofdm
        self.chunk = chunk
        sym_p, sc_p = pattern.positions()
        sym_g, sc_g = np.divmod(np.arange(S * K), K)
        # Offsets into the correlation tables r_f[dk + K - 1] and r_t[dm + S - 1].
        self.dk_hp = sc_g[:, None] - sc_p[None, :] + K - 1
        self.dm_hp = sym_g[:, None] - sym_p[None, :] + S - 1
        self.dk_pp = sc_p[:, None] - sc_p[None, :] + K - 1
        self.dm_pp = sym_p[:, None] - sym_p[None, :] + S - 1
        self.delta_k = np.arange(-(K - 1), K)
        self.delta_m = np.arange(-(S - 1), S)

    def __call__(self, slots: SlotBatch) -> np.ndarray:
        B = slots.rx_grid.shape[0]
        S, K = self.pattern.shape
        P = self.pattern.n_pilots
        h_p = ls_at_pilots(slots.rx_grid, self.pattern)
        r_f = frequency_correlation(slots.pdp, self.delta_k, self.ofdm.n_fft)
        r_t = time_correlation(slots.doppler_hz, self.delta_m, self.ofdm.symbol_duration_s)
        # Small diagonal loading keeps the solve stable in the noiseless case.
        loading = np.maximum(slots.noise_var, 1e-6)

        out = np.empty((B, S * K), dtype=np.complex128)
        for start in range(0, B, self.chunk):
            sl = slice(start, start + self.chunk)
            r_hp = r_f[sl][:, self.dk_hp] * r_t[sl][:, self.dm_hp]  # (b, S*K, P)
            r_pp = r_f[sl][:, self.dk_pp] * r_t[sl][:, self.dm_pp]  # (b, P, P)
            r_pp = r_pp + loading[sl, None, None] * np.eye(P)
            coeff = np.linalg.solve(r_pp, h_p[sl][..., None])  # (b, P, 1)
            out[sl] = (r_hp @ coeff)[..., 0]
        return out.reshape(B, S, K)


class PerfectCSI:
    """Genie: returns the true channel. Reference for BER without estimation error."""

    def __call__(self, slots: SlotBatch) -> np.ndarray:
        return slots.h_true


def baseline_estimators(pattern: PilotPattern, ofdm: OFDMConfig) -> dict[str, Estimator]:
    return {
        "ls_linear": InterpolationEstimator(pattern, "linear"),
        "ls_spline": InterpolationEstimator(pattern, "spline"),
        "lmmse": LMMSEEstimator(pattern, ofdm),
        "perfect": PerfectCSI(),
    }
