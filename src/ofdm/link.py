"""End-to-end link: bits -> QAM -> resource grid -> OFDM -> channel -> AWGN -> FFT -> estimate.

The steps are split (transmit / propagate / receive) so that one channel realization can
be reused for every SNR point of a sweep, which makes curves smoother and evaluation faster.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from . import ofdm as ofdm_mod
from .channel import (
    ChannelProfiles,
    add_awgn,
    apply_channel,
    frequency_response,
    generate_taps,
    sample_profiles,
)
from .config import Config, Region
from .estimators import Estimator
from .modulation import bits_per_symbol, demodulate, modulate
from .resource_grid import make_pilot_pattern


@dataclass(frozen=True)
class SlotBatch:
    """Everything known about a batch of B slots after reception."""

    bits: np.ndarray  # (B, n_data * q) transmitted data bits
    tx_grid: np.ndarray  # (B, S, K) transmitted resource grid (data + pilots)
    rx_grid: np.ndarray  # (B, S, K) received resource grid after the FFT
    h_true: np.ndarray  # (B, S, K) true channel per resource element
    noise_var: np.ndarray  # (B,) noise variance per resource element (= 1 / SNR)
    pdp: np.ndarray  # (B, L) power delay profile of each slot
    delay_spread_s: np.ndarray  # (B,)
    doppler_hz: np.ndarray  # (B,)

    @property
    def snr_db(self) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return -10 * np.log10(self.noise_var)


class LinkSimulator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.ofdm = cfg.ofdm
        self.pattern = make_pilot_pattern(cfg.ofdm, cfg.pilots)
        self.q = bits_per_symbol(cfg.ofdm.modulation)

    @property
    def bits_per_slot(self) -> int:
        return self.pattern.n_data * self.q

    def transmit(self, batch: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        """Return (bits, tx_grid, tx_signal)."""
        S, K = self.pattern.shape
        bits = rng.integers(0, 2, size=(batch, self.bits_per_slot), dtype=np.uint8)
        grid = np.broadcast_to(self.pattern.values, (batch, S, K)).copy()
        grid[:, ~self.pattern.mask] = modulate(bits, self.ofdm.modulation)
        return bits, grid, ofdm_mod.modulate(grid, self.ofdm)

    def propagate(
        self, tx_signal: np.ndarray, profiles: ChannelProfiles | None, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (noiseless rx_signal, h_true, pdp). profiles=None means pure AWGN."""
        B = tx_signal.shape[0]
        if profiles is None:
            S, K = self.pattern.shape
            return tx_signal, np.ones((B, S, K), dtype=np.complex128), np.ones((B, 1))
        realization = generate_taps(profiles, self.ofdm, self.cfg.channel, rng)
        rx = apply_channel(tx_signal, realization)
        return rx, frequency_response(realization, self.ofdm), realization.pdp

    def receive(
        self, rx_signal: np.ndarray, snr_db: float | np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Add noise and demodulate. snr_db may be a scalar or one value per slot."""
        B = rx_signal.shape[0]
        noise_var = np.broadcast_to(10 ** (-np.asarray(snr_db, dtype=float) / 10), (B,)).copy()
        if np.any(noise_var > 0):
            rx_signal = add_awgn(rx_signal, noise_var, rng)
        return ofdm_mod.demodulate(rx_signal, self.ofdm), noise_var

    def simulate(
        self,
        batch: int,
        snr_db: float | np.ndarray,
        rng: np.random.Generator,
        profiles: ChannelProfiles | None = None,
    ) -> SlotBatch:
        bits, tx_grid, tx_signal = self.transmit(batch, rng)
        rx_signal, h_true, pdp = self.propagate(tx_signal, profiles, rng)
        rx_grid, noise_var = self.receive(rx_signal, snr_db, rng)
        return _assemble(bits, tx_grid, rx_grid, h_true, noise_var, pdp, profiles)

    def detect(self, slots: SlotBatch, h_hat: np.ndarray) -> np.ndarray:
        """One-tap zero-forcing equalization on the data REs, then hard demapping."""
        data = ~self.pattern.mask
        x_hat = slots.rx_grid[:, data] / h_hat[:, data]
        return demodulate(x_hat, self.ofdm.modulation)

    def bit_errors(self, slots: SlotBatch, h_hat: np.ndarray) -> int:
        return int(np.count_nonzero(self.detect(slots, h_hat) != slots.bits))


def _assemble(
    bits: np.ndarray,
    tx_grid: np.ndarray,
    rx_grid: np.ndarray,
    h_true: np.ndarray,
    noise_var: np.ndarray,
    pdp: np.ndarray,
    profiles: ChannelProfiles | None,
) -> SlotBatch:
    B = bits.shape[0]
    zeros = np.zeros(B)
    return SlotBatch(
        bits=bits,
        tx_grid=tx_grid,
        rx_grid=rx_grid,
        h_true=h_true,
        noise_var=noise_var,
        pdp=pdp,
        delay_spread_s=profiles.delay_spread_s if profiles is not None else zeros,
        doppler_hz=profiles.doppler_hz if profiles is not None else zeros,
    )


def evaluate_grid(
    sim: LinkSimulator,
    estimators: Mapping[str, Estimator],
    delay_spreads_us: Iterable[float],
    dopplers_hz: Iterable[float],
    snr_db: Iterable[float],
    n_slots: int,
    batch_size: int,
    rng: np.random.Generator,
) -> list[dict[str, float | str]]:
    """evaluate() for every fixed (delay spread, Doppler) pair; both are added to each row."""
    snr_db, dopplers_hz = list(snr_db), list(dopplers_hz)
    rows: list[dict[str, float | str]] = []
    for tau in delay_spreads_us:
        for fd in dopplers_hz:
            region = Region((tau, tau), (fd, fd))
            for row in evaluate(sim, estimators, region, snr_db, n_slots, batch_size, rng):
                rows.append({"delay_spread_us": float(tau), "doppler_hz": float(fd), **row})
    return rows


def evaluate(
    sim: LinkSimulator,
    estimators: Mapping[str, Estimator],
    region: Region | None,
    snr_db: Iterable[float],
    n_slots: int,
    batch_size: int,
    rng: np.random.Generator,
) -> list[dict[str, float | str]]:
    """Monte-Carlo NMSE and BER per (SNR, estimator). region=None evaluates pure AWGN."""
    snr_db = list(snr_db)
    # accumulators: [error energy, channel energy, bit errors, bits]
    acc = {(snr, name): np.zeros(4) for snr in snr_db for name in estimators}
    for start in range(0, n_slots, batch_size):
        b = min(batch_size, n_slots - start)
        profiles = sample_profiles(region, b, rng) if region is not None else None
        bits, tx_grid, tx_signal = sim.transmit(b, rng)
        rx_signal, h_true, pdp = sim.propagate(tx_signal, profiles, rng)
        for snr in snr_db:
            rx_grid, noise_var = sim.receive(rx_signal, snr, rng)
            slots = _assemble(bits, tx_grid, rx_grid, h_true, noise_var, pdp, profiles)
            for name, est in estimators.items():
                h_hat = est(slots)
                a = acc[(snr, name)]
                a[0] += np.sum(np.abs(h_hat - h_true) ** 2)
                a[1] += np.sum(np.abs(h_true) ** 2)
                a[2] += sim.bit_errors(slots, h_hat)
                a[3] += bits.size

    rows: list[dict[str, float | str]] = []
    for (snr, name), (err, energy, bit_err, n_bits) in acc.items():
        nmse = err / energy
        with np.errstate(divide="ignore"):
            nmse_db = float(10 * np.log10(nmse))
        rows.append(
            {
                "snr_db": float(snr),
                "estimator": name,
                "nmse": float(nmse),
                "nmse_db": nmse_db,
                "ber": float(bit_err / n_bits),
                "bit_errors": int(bit_err),
                "n_bits": int(n_bits),
            }
        )
    return rows
