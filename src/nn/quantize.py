"""Quantize the exported correction network to int8 with ONNX Runtime, without new training.

Every value is rounded to one of 256 steps. Weights get a step size per filter, activations a step
size per tensor, calibrated on real network inputs. The calibration range decides how fine the steps
are: rare outliers from noisy slots would stretch it and make fine corrections at high SNR
impossible, clipping too much hurts at low SNR. scripts/compare_quantization.py measures this.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from ofdm import Config, LinkSimulator

from .config import DataConfig
from .dataset import make_dataset

CALIBRATION_SLOTS = 2000
PERCENTILE = 99.95  # smallest largest NMSE loss on validation slots, compare_quantization.py


def calibration_input(
    sim: LinkSimulator, cfg: Config, data: DataConfig, n_slots: int = CALIBRATION_SLOTS
) -> np.ndarray:
    """Network inputs from the validation seed. Test data must stay out of the calibration."""
    x, _ = make_dataset(sim, cfg.regions[data.region], n_slots, data.snr_db, data.val.seed)
    return x


class _SingleBatch(CalibrationDataReader):
    """All slots in one batch: the percentile calibration fails on batches of different size."""

    def __init__(self, input_name: str, x: np.ndarray) -> None:
        self.batches = iter([{input_name: x}])

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self.batches, None)


def quantize(
    onnx_path: str | Path,
    out_path: str | Path,
    calibration: np.ndarray,
    percentile: float | None = PERCENTILE,
    per_channel: bool = True,
    reduce_range: bool = True,
) -> None:
    """Write an int8 copy of an ONNX model, calibrated on the network inputs in calibration.

    percentile is the share of activation values inside the int8 range. ONNX Runtime clips the rest
    evenly on both sides, so 99.95 clips 0.025 % at each end. None covers all values (min/max).

    reduce_range limits the weights to 7 bits. x86 CPUs without VNNI add pairs of products in 16
    bits, where 8-bit weights can overflow: the same file then gives different results on such a
    CPU. With 7 bits the sum of two products fits, at most 2 * 255 * 64 = 32640.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    method = CalibrationMethod.MinMax if percentile is None else CalibrationMethod.Percentile
    extra = {"CalibTensorRangeSymmetric": False}
    if percentile is not None:
        extra["CalibPercentile"] = percentile
    with tempfile.TemporaryDirectory() as tmp:
        prepared = Path(tmp) / "prepared.onnx"
        quant_pre_process(onnx_path, prepared, skip_symbolic_shape=True)
        input_name = onnx.load(prepared).graph.input[0].name
        with contextlib.redirect_stdout(io.StringIO()):  # the calibration prints its progress
            quantize_static(
                prepared,
                out_path,
                _SingleBatch(input_name, calibration),
                quant_format=QuantFormat.QDQ,
                per_channel=per_channel,
                reduce_range=reduce_range,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=method,
                extra_options=extra,
            )


def output_step(onnx_path: str | Path) -> float:
    """Size of one int8 step at the output of a quantized model."""
    graph = onnx.load(onnx_path).graph
    node = next(n for n in graph.node if graph.output[0].name in n.output)
    step = next(i for i in graph.initializer if i.name == node.input[1])
    return float(numpy_helper.to_array(step))


def output_fingerprint(onnx_path: str | Path) -> str:
    """Hash of the int8 output for a fixed random input. Must not depend on the CPU."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(33).standard_normal((4, 3, 14, 72)).astype(np.float32)
    y = session.run(None, {session.get_inputs()[0].name: x})[0]
    steps = np.rint(y / output_step(onnx_path)).astype(np.int16)
    return hashlib.sha256(steps.tobytes()).hexdigest()[:16]


def ops_that_run(onnx_path: str | Path) -> dict[str, int]:
    """Operators ONNX Runtime executes after its optimizations, counted per type.

    The file itself only shows Conv between quantize and dequantize nodes. Whether that really runs
    in integers shows only here: QLinearConv does, a remaining Conv calculates in floating point.
    """
    with tempfile.TemporaryDirectory() as tmp:
        optimized = Path(tmp) / "optimized.onnx"
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.optimized_model_filepath = str(optimized)
        options.log_severity_level = 3  # warns that the result is specific to this CPU; it is
        ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
        return dict(Counter(node.op_type for node in onnx.load(optimized).graph.node))
