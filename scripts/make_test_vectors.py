"""Write inputs and reference outputs for the C++ tests. ONNX Runtime in Python is the reference.

Files in cpp/testdata/, all raw float32, little endian, in C order:
    input.bin      (slots, 3, 14, 72), one slot from region A per SNR in SNR_DB, then the first slot
                   again with doubled values, so that the int8 models have to clip their input
    <model>.bin    (slots, 2, 14, 72), the correction from models/<model>.onnx for each slot

Run this again after scripts/export_onnx.py has changed a model.

Example:
    python scripts/make_test_vectors.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nn.dataset import to_input
from nn.export import run_slot_by_slot
from ofdm import LinkSimulator, load_config, sample_profiles
from ofdm.estimators import InterpolationEstimator

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("cnn_A_log_nmse", "cnn_A_log_nmse_int8", "cnn_A_small", "cnn_A_small_int8")
SNR_DB = (0.0, 10.0, 20.0, 30.0)
SEED = 105  # independent of the training, validation and test slots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path, default=ROOT / "cpp" / "testdata")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim = LinkSimulator(cfg)
    interp = InterpolationEstimator(sim.pattern, "linear")
    rng = np.random.default_rng(SEED)
    inputs = []
    for snr in SNR_DB:
        slots = sim.simulate(1, snr, rng, sample_profiles(cfg.regions["A"], 1, rng))
        inputs.append(to_input(interp.interpolate(slots.rx_grid), slots.snr_db))
    doubled = inputs[0].copy()
    doubled[:, :2] *= 2
    x = np.concatenate([*inputs, doubled])

    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {"input": x}
    for name in MODELS:
        outputs[name] = run_slot_by_slot(args.models / f"{name}.onnx", x)
    for name, values in outputs.items():
        path = args.out / f"{name}.bin"
        values.astype("<f4").tofile(path)
        print(f"{path.name}: shape {values.shape}, {path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
