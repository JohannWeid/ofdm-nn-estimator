"""Export the trained CNN to ONNX, so that it can run without PyTorch.

Only the correction network is exported: input (B, 3, S, K) as built by nn.dataset.to_input,
output (B, 2, S, K). The caller adds the correction to the LS + linear estimate. That estimate
exists in the receiver anyway, and keeping it out of the model also keeps it out of the later
int8 quantization, where rounding it would cost accuracy at high SNR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .model import ResidualCNN

INPUT_NAME = "input"
OUTPUT_NAME = "correction"


def export_onnx(
    model: ResidualCNN,
    path: str | Path,
    n_symbols: int = 14,
    n_subcarriers: int = 72,
    opset: int = 18,
) -> None:
    """Write the correction network to one self-contained ONNX file; the batch size stays free."""
    if not model.use_snr:
        raise ValueError("only models with the SNR plane can be exported, see nn.dataset.to_input")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.body.eval(),
        (torch.zeros(1, 3, n_symbols, n_subcarriers),),
        path,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        opset_version=opset,
        external_data=False,
        dynamo=True,
        verbose=False,
    )


def max_deviation(model: ResidualCNN, onnx_path: str | Path, x: np.ndarray) -> float:
    """Largest absolute difference between the PyTorch and the ONNX Runtime output for x."""
    with torch.inference_mode():
        expected = model.body.eval()(torch.from_numpy(x)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {INPUT_NAME: x})[0]
    return float(np.abs(actual - expected).max())
