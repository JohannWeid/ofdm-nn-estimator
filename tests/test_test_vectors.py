"""The reference data for the C++ tests has to belong to the committed models (Phase 4, step 3)."""

import tomllib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from onnx import numpy_helper  # noqa: E402

from nn.export import INPUT_NAME, run_slot_by_slot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "cpp" / "testdata"
MODELS = ROOT / "models"
MODEL_NAMES = ["cnn_A_log_nmse", "cnn_A_small"]
SNR_DB = [0.0, 10.0, 20.0, 30.0]
N_SLOTS = len(SNR_DB) + 1
MAX_DEVIATION = 1e-5


def read(name: str, channels: int) -> np.ndarray:
    return np.fromfile(TESTDATA / f"{name}.bin", dtype="<f4").reshape(-1, channels, 14, 72)


def test_cpp_and_python_use_the_same_onnx_runtime() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pinned = next(d for d in project["optional-dependencies"]["deploy"] if "onnxruntime" in d)
    version = pinned.split("==")[1]
    cmake = (ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert f"onnxruntime-linux-x64-{version}.tgz" in cmake


def test_input_holds_one_slot_per_snr_and_a_doubled_one() -> None:
    x = read("input", 3)
    assert x.shape == (N_SLOTS, 3, 14, 72)
    for slot, snr in zip(x[:-1], SNR_DB, strict=True):
        assert np.allclose(slot[2], snr / 30)
        assert np.ptp(slot[:2]) > 0.5  # a real channel estimate, not an empty grid
    assert np.array_equal(x[-1, :2], 2 * x[0, :2])
    assert np.array_equal(x[-1, 2], x[0, 2])


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_last_slot_makes_the_int8_models_clip_their_input(name: str) -> None:
    graph = onnx.load(MODELS / f"{name}_int8.onnx").graph
    node = next(n for n in graph.node if n.op_type == "QuantizeLinear" and n.input[0] == INPUT_NAME)
    step = next(i for i in graph.initializer if i.name == node.input[1])
    limit = 127 * float(numpy_helper.to_array(step))
    x = read("input", 3)
    assert np.abs(x[-1, :2]).max() > limit
    assert np.abs(x[:-1, :2]).max() < limit


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fp32_vectors_belong_to_the_model(name: str) -> None:
    """Fails if a model changes without running scripts/make_test_vectors.py again."""
    expected = read(name, 2)
    assert expected.shape == (N_SLOTS, 2, 14, 72)
    actual = run_slot_by_slot(MODELS / f"{name}.onnx", read("input", 3))
    assert np.abs(actual - expected).max() < MAX_DEVIATION


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_int8_vectors_are_bit_exact_on_this_cpu(name: str) -> None:
    """The vectors come from a laptop with VNNI, the CI runner has none."""
    expected = read(f"{name}_int8", 2)
    assert expected.shape == (N_SLOTS, 2, 14, 72)
    actual = run_slot_by_slot(MODELS / f"{name}_int8.onnx", read("input", 3))
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_int8_vectors_are_not_the_fp32_ones(name: str) -> None:
    fp32, int8 = read(name, 2), read(f"{name}_int8", 2)
    assert np.abs(fp32 - int8).max() > MAX_DEVIATION
    assert len(np.unique(int8)) <= 256  # int8 has only so many values, FP32 far more
    assert len(np.unique(fp32)) > 1000
