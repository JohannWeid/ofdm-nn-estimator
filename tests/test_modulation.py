import numpy as np
import pytest

from ofdm.modulation import bits_per_symbol, constellation, demodulate, modulate


@pytest.mark.parametrize("modulation", ["qpsk", "16qam"])
def test_roundtrip(modulation: str, rng: np.random.Generator) -> None:
    bits = rng.integers(0, 2, size=(3, 400 * bits_per_symbol(modulation)), dtype=np.uint8)
    assert np.array_equal(demodulate(modulate(bits, modulation), modulation), bits)


@pytest.mark.parametrize("modulation", ["qpsk", "16qam"])
def test_unit_average_energy(modulation: str) -> None:
    points = constellation(modulation)
    assert points.size == 2 ** bits_per_symbol(modulation)
    assert np.mean(np.abs(points) ** 2) == pytest.approx(1.0)


@pytest.mark.parametrize("modulation", ["qpsk", "16qam"])
def test_gray_neighbours_differ_in_one_bit(modulation: str) -> None:
    q = bits_per_symbol(modulation)
    labels = ((np.arange(2**q)[:, None] >> np.arange(q - 1, -1, -1)) & 1).astype(np.uint8)
    points = modulate(labels, modulation).ravel()
    d = np.abs(points[:, None] - points[None, :])
    d_min = d[d > 1e-9].min()
    for i, j in zip(*np.nonzero(np.isclose(d, d_min)), strict=True):
        assert np.sum(labels[i] != labels[j]) == 1
