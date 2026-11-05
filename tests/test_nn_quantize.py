"""Tests for the int8 quantization (Phase 4, step 2)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxscript")

from onnx import numpy_helper  # noqa: E402

from nn.config import load_nn_config  # noqa: E402
from nn.estimator import OnnxEstimator  # noqa: E402
from nn.export import INPUT_NAME  # noqa: E402
from nn.quantize import (  # noqa: E402
    calibration_input,
    ops_that_run,
    output_fingerprint,
    output_step,
    quantize,
)
from ofdm import LinkSimulator, load_config, sample_profiles  # noqa: E402
from ofdm.config import Config  # noqa: E402
from ofdm.estimators import InterpolationEstimator  # noqa: E402
from ofdm.metrics import nmse, to_db  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
STANDARD = MODELS / "cnn_A_log_nmse.onnx"
MODEL_NAMES = ["cnn_A_log_nmse", "cnn_A_small"]
N_LAYERS = {"cnn_A_log_nmse": 4, "cnn_A_small": 3}
FINGERPRINTS = {"cnn_A_log_nmse": "9f598b86a12068a2", "cnn_A_small": "4607b853918f6350"}


def initializers(path: Path) -> dict[str, np.ndarray]:
    return {i.name: numpy_helper.to_array(i) for i in onnx.load(path).graph.initializer}


def activation_steps(path: Path) -> dict[str, float]:
    """Step size of every activation, input and output included."""
    values = initializers(path)
    return {
        node.input[0]: float(values[node.input[1]])
        for node in onnx.load(path).graph.node
        if node.op_type == "QuantizeLinear"
    }


def int8_weights(path: Path) -> list[np.ndarray]:
    """The convolution filters as stored in the file, in graph order."""
    values = initializers(path)
    return [v for v in values.values() if v.dtype == np.int8 and v.ndim == 4]


def run(path: Path, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {INPUT_NAME: x})[0]


@pytest.fixture(scope="module")
def x_cal() -> np.ndarray:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    return calibration_input(LinkSimulator(cfg), cfg, load_nn_config(ROOT / "configs/nn.yaml").data)


@pytest.fixture(scope="module")
def quantized(tmp_path_factory: pytest.TempPathFactory, x_cal: np.ndarray) -> Path:
    path = tmp_path_factory.mktemp("int8") / "int8.onnx"
    quantize(STANDARD, path, x_cal)
    return path


def test_all_convolutions_run_in_integers(quantized: Path) -> None:
    ops = ops_that_run(quantized)
    assert ops["QLinearConv"] == 4
    assert "Conv" not in ops


def test_ops_that_run_sees_a_float_model() -> None:
    """Otherwise the test above could not fail."""
    ops = ops_that_run(STANDARD)
    assert "QLinearConv" not in ops
    assert ops.get("Conv", 0) + ops.get("FusedConv", 0) == 4


def test_every_filter_has_its_own_weight_step(quantized: Path) -> None:
    values = initializers(quantized)
    weight_nodes = [
        node
        for node in onnx.load(quantized).graph.node
        if node.op_type == "DequantizeLinear" and values.get(node.input[0], np.zeros(0)).ndim == 4
    ]
    assert len(weight_nodes) == 4
    for node in weight_nodes:
        assert values[node.input[0]].dtype == np.int8
        assert values[node.input[1]].size == values[node.input[0]].shape[0]


def test_range_ignores_rare_outliers(quantized: Path, x_cal: np.ndarray, tmp_path: Path) -> None:
    min_max = tmp_path / "min_max.onnx"
    quantize(STANDARD, min_max, x_cal, percentile=None)
    assert output_step(quantized) < 0.8 * output_step(min_max)


def test_rectified_activations_use_the_full_range(quantized: Path) -> None:
    """A ReLU output is never negative, so all 256 steps must lie above zero."""
    graph = onnx.load(quantized).graph
    values = initializers(quantized)
    relu_nodes = [n for n in graph.node if n.op_type == "QuantizeLinear" and "relu" in n.input[0]]
    assert len(relu_nodes) == 3
    for node in relu_nodes:
        assert values[node.input[2]] == -128


def test_batch_size_stays_free(quantized: Path, x_cal: np.ndarray) -> None:
    for n_slots in (1, 5):
        assert run(quantized, x_cal[:n_slots]).shape == (n_slots, 2, 14, 72)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_committed_int8_files_match_the_code(name: str, x_cal: np.ndarray, tmp_path: Path) -> None:
    """Fails if a model or the calibration changes without running scripts/export_onnx.py again."""
    committed = MODELS / f"{name}_int8.onnx"
    fresh = tmp_path / "fresh.onnx"
    quantize(MODELS / f"{name}.onnx", fresh, x_cal)
    steps, fresh_steps = activation_steps(committed), activation_steps(fresh)
    assert steps.keys() == fresh_steps.keys()
    for tensor, step in steps.items():
        assert step == pytest.approx(fresh_steps[tensor], rel=0.01)
    weights, fresh_weights = int8_weights(committed), int8_weights(fresh)
    assert len(weights) == N_LAYERS[name]
    for w, fresh_w in zip(weights, fresh_weights, strict=True):
        assert np.abs(w.astype(int) - fresh_w.astype(int)).max() <= 1
    ops = ops_that_run(committed)
    assert ops["QLinearConv"] == N_LAYERS[name]
    assert "Conv" not in ops


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_int8_output_does_not_depend_on_the_cpu(name: str) -> None:
    """The values come from a laptop with VNNI; the CI runner checks them on its own CPU.

    8-bit weights fail this on x86 without VNNI. After retraining, scripts/export_onnx.py prints the
    new values.
    """
    assert output_fingerprint(MODELS / f"{name}_int8.onnx") == FINGERPRINTS[name]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_two_products_fit_into_16_bits(name: str) -> None:
    """x86 without VNNI adds pairs of u8 * s8 products in 16 bits, see nn.quantize.quantize."""
    for weights in int8_weights(MODELS / f"{name}_int8.onnx"):
        assert 2 * 255 * np.abs(weights.astype(int)).max() <= 32767


def test_eight_bit_weights_do_not_fit(x_cal: np.ndarray, tmp_path: Path) -> None:
    """Otherwise the test above could not fail."""
    path = tmp_path / "eight_bit.onnx"
    quantize(STANDARD, path, x_cal, reduce_range=False)
    assert 2 * 255 * max(np.abs(w.astype(int)).max() for w in int8_weights(path)) > 32767


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_committed_int8_files_stay_close_to_fp32(name: str, x_cal: np.ndarray) -> None:
    """Measured: 0.020 and 0.027. A second training run of the same model differs by 0.035."""
    x = x_cal[:100]
    error = run(MODELS / f"{name}_int8.onnx", x) - run(MODELS / f"{name}.onnx", x)
    assert np.sqrt(np.mean(error**2)) < 0.03


@pytest.mark.parametrize("snr_db", [0.0, 30.0])
def test_int8_keeps_the_gain_over_ls(sim: LinkSimulator, cfg: Config, snr_db: float) -> None:
    """A quick version of the acceptance from scripts/compare_quantization.py, on fewer slots."""
    rng = np.random.default_rng(32)
    slots = sim.simulate(64, snr_db, rng, sample_profiles(cfg.regions["A"], 64, rng))
    ls, fp32, int8 = (
        float(to_db(nmse(estimator(slots), slots.h_true)))
        for estimator in (
            InterpolationEstimator(sim.pattern, "linear"),
            OnnxEstimator(STANDARD, sim.pattern),
            OnnxEstimator(MODELS / "cnn_A_log_nmse_int8.onnx", sim.pattern),
        )
    )
    assert int8 < ls - 1.0
    assert int8 < fp32 + 0.6
