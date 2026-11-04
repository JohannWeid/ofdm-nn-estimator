"""Tests for the ONNX export (Phase 4, step 1)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")
pytest.importorskip("onnxscript")

from nn.config import ModelConfig  # noqa: E402
from nn.estimator import NeuralEstimator, OnnxEstimator  # noqa: E402
from nn.export import INPUT_NAME, export_onnx, max_deviation  # noqa: E402
from nn.model import build_model, load_model  # noqa: E402
from ofdm import LinkSimulator, sample_profiles  # noqa: E402
from ofdm.config import Config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAX_DEVIATION = 1e-5


def random_input(n_slots: int, seed: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_slots, 3, 14, 72)).astype(np.float32)


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("onnx") / "cnn.onnx"
    model = load_model(ROOT / "models" / "cnn_A_log_nmse.pt")
    export_onnx(model, path)
    return model, path


def test_export_matches_pytorch(exported) -> None:
    model, path = exported
    assert max_deviation(model, path, random_input(4)) < MAX_DEVIATION


def test_batch_size_is_free(exported) -> None:
    _, path = exported
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for n_slots in (1, 5):
        output = session.run(None, {INPUT_NAME: random_input(n_slots)})[0]
        assert output.shape == (n_slots, 2, 14, 72)


def test_export_is_one_self_contained_file(exported) -> None:
    """C++ and the browser demo load a single file; weights must not sit next to it."""
    _, path = exported
    assert list(path.parent.iterdir()) == [path]
    assert path.stat().st_size > 5378 * 4  # all parameters as float32 are inside


def test_onnx_estimator_matches_neural_estimator(
    sim: LinkSimulator, cfg: Config, exported
) -> None:
    model, path = exported
    rng = np.random.default_rng(31)
    slots = sim.simulate(8, 10.0, rng, sample_profiles(cfg.regions["A"], 8, rng))
    h_onnx = OnnxEstimator(path, sim.pattern)(slots)
    h_torch = NeuralEstimator(model, sim.pattern)(slots)
    assert np.iscomplexobj(h_onnx)
    assert np.abs(h_onnx - h_torch).max() < MAX_DEVIATION


def test_whole_model_export_includes_the_addition(exported, tmp_path: Path) -> None:
    model, _ = exported
    path = tmp_path / "whole.onnx"
    export_onnx(model, path, whole_model=True)
    x = random_input(2)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model.eval()(torch.from_numpy(x)).numpy()
    assert np.abs(session.run(None, {INPUT_NAME: x})[0] - expected).max() < MAX_DEVIATION


def test_model_without_snr_plane_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SNR plane"):
        export_onnx(build_model(ModelConfig(use_snr=False)), tmp_path / "cnn.onnx")


@pytest.mark.parametrize("name", ["cnn_A_log_nmse", "cnn_A_small"])
def test_committed_onnx_files_belong_to_the_committed_weights(name: str) -> None:
    """Fails if a model is retrained without running scripts/export_onnx.py again."""
    model = load_model(ROOT / "models" / f"{name}.pt")
    assert max_deviation(model, ROOT / "models" / f"{name}.onnx", random_input(4)) < MAX_DEVIATION
