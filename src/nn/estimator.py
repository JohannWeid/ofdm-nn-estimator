"""The trained network as a channel estimator, interchangeable with the classical ones."""

from __future__ import annotations

import numpy as np
import torch

from ofdm import SlotBatch
from ofdm.estimators import InterpolationEstimator
from ofdm.resource_grid import PilotPattern

from .dataset import to_input
from .train import predict


class NeuralEstimator:
    """LS + linear interpolation, refined by the CNN. Fulfils ofdm.estimators.Estimator."""

    def __init__(self, model: torch.nn.Module, pattern: PilotPattern) -> None:
        self.model = model
        self.interp = InterpolationEstimator(pattern, "linear")

    def __call__(self, slots: SlotBatch) -> np.ndarray:
        x = to_input(self.interp.interpolate(slots.rx_grid), slots.snr_db)
        y = predict(self.model, torch.from_numpy(x)).numpy()
        return y[:, 0].astype(np.float64) + 1j * y[:, 1].astype(np.float64)
