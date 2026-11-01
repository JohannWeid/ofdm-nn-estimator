"""Tests for the CNN (Phase 3, step 2). Skipped when torch is not installed."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nn.dataset import make_dataset  # noqa: E402
from nn.model import ResidualCNN, count_parameters, macs_per_slot  # noqa: E402
from ofdm import LinkSimulator  # noqa: E402
from ofdm.config import Config  # noqa: E402

S, K = 14, 72


def test_output_shape_and_dtype() -> None:
    y = ResidualCNN()(torch.randn(5, 3, S, K))
    assert y.shape == (5, 2, S, K)
    assert y.dtype == torch.float32


def test_untrained_model_returns_ls_estimate() -> None:
    x = torch.randn(4, 3, S, K)
    assert torch.equal(ResidualCNN()(x), x[:, :2])


def test_without_snr_input_the_snr_plane_is_ignored() -> None:
    torch.manual_seed(0)
    model = ResidualCNN(use_snr=False)
    torch.nn.init.normal_(model.body[-1].weight)  # make the correction non-zero
    x = torch.randn(2, 3, S, K)
    x_other_snr = x.clone()
    x_other_snr[:, 2] = 0.9
    assert torch.equal(model(x), model(x_other_snr))


def test_size_of_default_model() -> None:
    model = ResidualCNN()
    assert count_parameters(model) == 5378
    assert macs_per_slot(model) == 5328 * S * K


def test_fully_convolutional_on_other_grid_sizes() -> None:
    assert ResidualCNN()(torch.randn(1, 3, 7, 36)).shape == (1, 2, 7, 36)


def test_a_few_training_steps_beat_the_ls_input(sim: LinkSimulator, cfg: Config) -> None:
    """The zero-initialized last layer must not block learning."""
    x, y = (torch.from_numpy(a) for a in make_dataset(sim, cfg.regions["A"], 64, (0, 10), 0))
    torch.manual_seed(0)
    model = ResidualCNN()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.MSELoss()
    initial = loss_fn(model(x), y).item()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.8 * initial
    assert np.isfinite(loss.item())
