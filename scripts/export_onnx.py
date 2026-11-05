"""Export trained models to ONNX, check them against PyTorch and quantize them to int8.

Examples:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model models/cnn_A_small.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nn.config import load_nn_config
from nn.dataset import make_dataset
from nn.export import export_onnx, max_deviation
from nn.model import load_model
from nn.quantize import calibration_input, ops_that_run, output_fingerprint, quantize
from ofdm import LinkSimulator, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [ROOT / "models" / "cnn_A_log_nmse.pt", ROOT / "models" / "cnn_A_small.pt"]
MAX_DEVIATION = 1e-5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--nn-config", type=Path, default=ROOT / "configs" / "nn.yaml")
    parser.add_argument("--model", type=Path, action="append", help="repeatable, .pt file")
    parser.add_argument("--check-slots", type=int, default=100, help="validation slots to compare")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = load_nn_config(args.nn_config).data
    sim = LinkSimulator(cfg)
    x, _ = make_dataset(
        sim, cfg.regions[data.region], args.check_slots, data.snr_db, data.val.seed
    )
    x_calibration = calibration_input(sim, cfg, data)

    for path in args.model or DEFAULT_MODELS:
        model = load_model(path)
        out = path.with_suffix(".onnx")
        export_onnx(model, out)
        deviation = max_deviation(model, out, x)
        print(f"{out.name}: {out.stat().st_size / 1024:.1f} KB, largest deviation from PyTorch "
              f"over {args.check_slots} slots: {deviation:.2e}")
        if deviation >= MAX_DEVIATION:
            raise SystemExit(f"{out.name}: deviation is not below {MAX_DEVIATION:g}")

        out_int8 = path.with_name(f"{path.stem}_int8.onnx")
        quantize(out, out_int8, x_calibration)
        ops = ops_that_run(out_int8)
        print(f"{out_int8.name}: {out_int8.stat().st_size / 1024:.1f} KB, operators: {ops}, "
              f"fingerprint for the tests: {output_fingerprint(out_int8)}")
        if "Conv" in ops:
            raise SystemExit(f"{out_int8.name}: a convolution still runs in floating point")


if __name__ == "__main__":
    main()
