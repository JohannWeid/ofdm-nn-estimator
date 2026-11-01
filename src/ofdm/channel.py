"""Time-varying tapped-delay-line (TDL) Rayleigh channel with exponential PDP, plus AWGN.

Tap l has delay l * Ts (sample-spaced) and average power p_l ~ exp(-l * Ts / tau),
normalized so that sum(p_l) = 1, i.e. E|H|^2 = 1 on every subcarrier.
Each tap gain evolves over the slot as a sum of sinusoids with random arrival angles
(Clarke's model), whose Doppler spectrum approximates the Jakes spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import j0

from .config import ChannelConfig, OFDMConfig, Region
from .ofdm import subcarrier_bins


@dataclass(frozen=True)
class ChannelProfiles:
    """Large-scale channel parameters per slot."""

    delay_spread_s: np.ndarray  # (B,) decay constant tau of the exponential PDP
    doppler_hz: np.ndarray  # (B,) maximum Doppler shift f_D

    def __len__(self) -> int:
        return len(self.delay_spread_s)


@dataclass(frozen=True)
class ChannelRealization:
    taps: np.ndarray  # (B, L, N) complex gain of tap l at output sample n
    pdp: np.ndarray  # (B, L) average tap powers (zero-padded to a common L)


def sample_profiles(region: Region, batch: int, rng: np.random.Generator) -> ChannelProfiles:
    tau = rng.uniform(*region.delay_spread_us, size=batch) * 1e-6
    fd = rng.uniform(*region.doppler_hz, size=batch)
    return ChannelProfiles(tau, fd)


def fixed_profiles(delay_spread_s: float, doppler_hz: float, batch: int) -> ChannelProfiles:
    return ChannelProfiles(np.full(batch, delay_spread_s), np.full(batch, doppler_hz))


def exponential_pdp(
    delay_spread_s: float, sample_period_s: float, max_taps: int, floor_db: float
) -> np.ndarray:
    """Sample-spaced exponential power delay profile, truncated at floor_db, sum = 1."""
    if delay_spread_s <= 0:
        return np.ones(1)
    span = delay_spread_s / sample_period_s * np.log(10 ** (-floor_db / 10))
    n_taps = min(max_taps, int(np.floor(span)) + 1)
    p = np.exp(-np.arange(n_taps) * sample_period_s / delay_spread_s)
    return p / p.sum()


def pdp_matrix(profiles: ChannelProfiles, ofdm: OFDMConfig, chan: ChannelConfig) -> np.ndarray:
    pdps = [
        exponential_pdp(tau, ofdm.sample_period_s, chan.max_taps, chan.pdp_floor_db)
        for tau in profiles.delay_spread_s
    ]
    out = np.zeros((len(pdps), max(len(p) for p in pdps)))
    for i, p in enumerate(pdps):
        out[i, : len(p)] = p
    return out


def generate_taps(
    profiles: ChannelProfiles,
    ofdm: OFDMConfig,
    chan: ChannelConfig,
    rng: np.random.Generator,
) -> ChannelRealization:
    pdp = pdp_matrix(profiles, ofdm, chan)
    B, L = pdp.shape
    M = chan.n_sinusoids
    S, SL = ofdm.n_symbols, ofdm.symbol_length

    alpha = rng.uniform(0, 2 * np.pi, size=(B, L, M))  # angle of arrival
    phi = rng.uniform(0, 2 * np.pi, size=(B, L, M))  # initial phase
    nu = profiles.doppler_hz[:, None, None] * np.cos(alpha)  # Doppler shift per path
    w = 2 * np.pi * nu * ofdm.sample_period_s  # phase increment per sample

    # h_l[n] = sum_m exp(j (w_m n + phi_m)). With n = s * SL + r the exponential factorizes
    # into a symbol part and a sample part, so the sum over m becomes a batched matmul.
    sym_part = np.exp(1j * (w[..., None, :] * (SL * np.arange(S))[:, None] + phi[..., None, :]))
    sample_part = np.exp(1j * w[..., :, None] * np.arange(SL))
    taps = (sym_part @ sample_part).reshape(B, L, S * SL)  # (B, L, S, M) @ (B, L, M, SL)
    taps *= np.sqrt(pdp / M)[..., None]
    return ChannelRealization(taps, pdp)


def apply_channel(signal: np.ndarray, realization: ChannelRealization) -> np.ndarray:
    """y[n] = sum_l h_l[n] x[n - l]  (samples before the slot are zero)."""
    taps = realization.taps
    out = taps[:, 0] * signal
    for lag in range(1, taps.shape[1]):
        out[:, lag:] += taps[:, lag, lag:] * signal[:, :-lag]
    return out


def frequency_response(realization: ChannelRealization, ofdm: OFDMConfig) -> np.ndarray:
    """True channel per resource element (B, S, K).

    Each tap is averaged over the FFT window of every OFDM symbol. For a channel that
    varies within a symbol this is the diagonal of the ICI matrix; the off-diagonal
    part (inter-carrier interference) acts as additional noise.
    """
    taps = realization.taps
    B, L, _ = taps.shape
    windows = taps.reshape(B, L, ofdm.n_symbols, ofdm.symbol_length)[..., ofdm.cp_length :]
    mean_taps = windows.mean(axis=-1)  # (B, L, S)
    steering = np.exp(-2j * np.pi * np.outer(np.arange(L), subcarrier_bins(ofdm)) / ofdm.n_fft)
    return np.einsum("bls,lk->bsk", mean_taps, steering)


def add_awgn(signal: np.ndarray, noise_var: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Complex white Gaussian noise with per-slot variance noise_var (B,)."""
    std = np.sqrt(np.asarray(noise_var) / 2)[:, None]
    noise = rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    return signal + std * noise


def frequency_correlation(pdp: np.ndarray, delta_k: np.ndarray, n_fft: int) -> np.ndarray:
    """E[H_k H_{k-dk}^*] = sum_l p_l exp(-j 2 pi dk l / N) for each slot: (B, len(delta_k))."""
    lags = np.arange(pdp.shape[1])
    return pdp @ np.exp(-2j * np.pi * np.outer(lags, delta_k) / n_fft)


def time_correlation(
    doppler_hz: np.ndarray, delta_m: np.ndarray, symbol_duration_s: float
) -> np.ndarray:
    """Jakes autocorrelation J0(2 pi f_D dm T_sym) for each slot: (B, len(delta_m))."""
    return j0(2 * np.pi * np.outer(doppler_hz, delta_m) * symbol_duration_s)
