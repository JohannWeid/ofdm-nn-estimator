"""The data of the browser demo has to belong to the committed models (Phase 5, step 1)."""

import json
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from nn.web_data import int8_layers, link_data  # noqa: E402
from ofdm.config import Config  # noqa: E402
from ofdm.resource_grid import make_pilot_pattern  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
WEB_MODEL = ROOT / "web" / "model"
STRUCTURE = {
    "cnn_A_log_nmse_int8": [(3, 16), (16, 16), (16, 16), (16, 2)],
    "cnn_A_small_int8": [(3, 8), (8, 8), (8, 2)],
}


def read_json(name: str) -> dict:
    return json.loads((WEB_MODEL / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", STRUCTURE)
def test_model_files_belong_to_the_committed_int8_models(name: str) -> None:
    """Fails if a model changes without running scripts/export_web_data.py again."""
    assert read_json(name) == int8_layers(MODELS / f"{name}.onnx")


@pytest.mark.parametrize("name", STRUCTURE)
def test_layers_have_the_expected_shape(name: str) -> None:
    layers = int8_layers(MODELS / f"{name}.onnx")["layers"]
    assert [(la["in_channels"], la["out_channels"]) for la in layers] == STRUCTURE[name]
    for layer in layers:
        assert len(layer["weights"]) == layer["out_channels"] * layer["in_channels"] * 9
        assert len(layer["weight_steps"]) == layer["out_channels"]
        assert len(layer["biases"]) == layer["out_channels"]
        assert max(abs(w) for w in layer["weights"]) <= 64  # 7 bits, see nn.quantize.quantize


def test_number_of_parameters_matches_the_models() -> None:
    def count(name: str) -> int:
        layers = int8_layers(MODELS / f"{name}.onnx")["layers"]
        return sum(len(la["weights"]) + len(la["biases"]) for la in layers)

    assert count("cnn_A_log_nmse_int8") == 5378
    assert count("cnn_A_small_int8") == 954


def test_link_file_matches_the_configuration(cfg: Config) -> None:
    """Fails if the configuration changes without running scripts/export_web_data.py again."""
    stored, fresh = read_json("link"), link_data(cfg)
    stored_pilots, fresh_pilots = stored.pop("pilots"), fresh.pop("pilots")
    assert stored == fresh
    assert stored_pilots["symbol"] == fresh_pilots["symbol"]
    assert stored_pilots["subcarrier"] == fresh_pilots["subcarrier"]
    # The pilots come from np.exp, whose last bit can differ between operating systems.
    assert np.allclose(stored_pilots["re"], fresh_pilots["re"], rtol=0, atol=1e-12)
    assert np.allclose(stored_pilots["im"], fresh_pilots["im"], rtol=0, atol=1e-12)


def test_pilots_are_in_the_order_of_the_least_squares_estimate(cfg: Config) -> None:
    pattern = make_pilot_pattern(cfg.ofdm, cfg.pilots)
    pilots = link_data(cfg)["pilots"]
    grid = np.zeros(pattern.shape, dtype=complex)
    values = np.array(pilots["re"]) + 1j * np.array(pilots["im"])
    grid[pilots["symbol"], pilots["subcarrier"]] = values
    assert np.array_equal(grid, pattern.values)
    assert np.array_equal(np.array(pilots["re"]), pattern.values[pattern.mask].real)


def test_float_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a quantized model"):
        int8_layers(MODELS / "cnn_A_log_nmse.onnx")


def test_other_kernel_size_is_rejected(tmp_path: Path) -> None:
    model = onnx.load(MODELS / "cnn_A_log_nmse_int8.onnx")
    conv = next(node for node in model.graph.node if node.op_type == "Conv")
    next(a for a in conv.attribute if a.name == "kernel_shape").ints[:] = [5, 5]
    path = tmp_path / "other.onnx"
    onnx.save(model, path)
    with pytest.raises(ValueError, match="unsupported convolution"):
        int8_layers(path)
