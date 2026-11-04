"""The trained network as a channel estimator, interchangeable with the classical ones."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ofdm import SlotBatch
from ofdm.estimators import InterpolationEstimator
from ofdm.resource_grid import PilotPattern

from .dataset import to_input
from .train import predict


def _to_complex(y: np.ndarray) -> np.ndarray:
    """(B, 2, S, K) float -> (B, S, K) complex."""
    return y[:, 0].astype(np.float64) + 1j * y[:, 1].astype(np.float64)


class NeuralEstimator:
    """LS + linear interpolation, refined by the CNN. Fulfils ofdm.estimators.Estimator."""

    def __init__(self, model: torch.nn.Module, pattern: PilotPattern) -> None:
        self.model = model
        self.interp = InterpolationEstimator(pattern, "linear")

    def __call__(self, slots: SlotBatch) -> np.ndarray:
        x = to_input(self.interp.interpolate(slots.rx_grid), slots.snr_db)
        return _to_complex(predict(self.model, torch.from_numpy(x)).numpy())


class OnnxEstimator:
    """Like NeuralEstimator, but the correction comes from an exported model in ONNX Runtime."""

    def __init__(self, onnx_path: str | Path, pattern: PilotPattern, threads: int = 0) -> None:
        import onnxruntime as ort  # optional dependency, see the "deploy" extra

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # 0 lets ONNX Runtime choose
        self.session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.interp = InterpolationEstimator(pattern, "linear")

    def __call__(self, slots: SlotBatch) -> np.ndarray:
        x = to_input(self.interp.interpolate(slots.rx_grid), slots.snr_db)
        correction = self.session.run(None, {self.input_name: x})[0]
        return _to_complex(x[:, :2] + correction)
