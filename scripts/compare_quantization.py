"""What does int8 cost? The int8 models against FP32, NMSE and BER over SNR.

Part "percentile" decides how much of the activation range the int8 steps cover. It runs on
validation slots, so that this choice does not touch the test data. Part "variants" compares the
committed int8 models from scripts/export_onnx.py on the test slots of the README, together with
the variants that show where the loss comes from. Variants with 8-bit weights can give other
results on an x86 CPU without VNNI, see nn.quantize.quantize.

Examples:
    python scripts/compare_quantization.py
    python scripts/compare_quantization.py --part variants --n-slots 500
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from nn.config import load_nn_config
from nn.dataset import to_input
from nn.estimator import OnnxEstimator
from nn.export import export_onnx
from nn.model import load_model
from nn.quantize import PERCENTILE, calibration_input, ops_that_run, output_step, quantize
from ofdm import LinkSimulator, SlotBatch, evaluate, load_config
from ofdm.estimators import InterpolationEstimator
from ofdm.metrics import snr_at_ber
from ofdm.plotting import write_csv

ROOT = Path(__file__).resolve().parents[1]
STANDARD = "cnn_A_log_nmse"
SMALL = "cnn_A_small"
PERCENTILES = (99.9, 99.95, 99.98, 99.99, 99.995)
BER_TARGETS = {"A": 1e-2, "B": 3e-2}


class WholeModelEstimator(OnnxEstimator):
    """The naive way: the addition is part of the model, so the estimate itself passes int8.

    Its output can be exactly zero, which the one-tap equalizer then divides by.
    """

    def __call__(self, slots: SlotBatch) -> np.ndarray:
        x = to_input(self.interp.interpolate(slots.rx_grid), slots.snr_db)
        y = self.session.run(None, {self.input_name: x})[0]
        return y[:, 0] + 1j * y[:, 1]


def print_table(rows: list[dict], reference: dict[str, str], target: float) -> None:
    """NMSE per SNR, largest NMSE loss against the reference estimator, SNR at the target BER."""
    snrs = sorted({r["snr_db"] for r in rows})
    nmse = {(r["estimator"], r["snr_db"]): r["nmse_db"] for r in rows}
    ber = {(r["estimator"], r["snr_db"]): r["ber"] for r in rows}
    print(f"{'':26s}" + "".join(f"{s:7.0f}" for s in snrs) + f"  largest loss  SNR@{target:g}")
    for name in dict.fromkeys(r["estimator"] for r in rows):
        line = f"{name:26s}" + "".join(f"{nmse[(name, s)]:7.2f}" for s in snrs)
        loss = ""
        if name in reference:
            loss = f"{max(nmse[(name, s)] - nmse[(reference[name], s)] for s in snrs):.2f}"
        crossing = snr_at_ber(snrs, [ber[(name, s)] for s in snrs], target)
        print(f"{line}{loss:>14s}{crossing:10.2f}")


def compare_percentiles(sim, cfg, data, x_cal, work: Path, n_slots: int, out: Path) -> None:
    fp32 = ROOT / "models" / f"{STANDARD}.onnx"
    estimators = {"FP32": OnnxEstimator(fp32, sim.pattern)}
    for percentile in PERCENTILES:
        path = work / f"p{percentile}.onnx"
        quantize(fp32, path, x_cal, percentile)
        estimators[f"{percentile:g} %"] = OnnxEstimator(path, sim.pattern)
    region = cfg.regions[data.region]
    rng = np.random.default_rng(data.val.seed)
    rows = evaluate(
        sim, estimators, region, cfg.evaluation.snr_db, n_slots, cfg.evaluation.batch_size, rng
    )
    write_csv(rows, out / "quantization_percentile.csv")

    print(f"Percentile, standard model, validation slots, region {data.region}, "
          f"{n_slots} slots per SNR; the code uses {PERCENTILE:g} %")
    print_table(rows, {f"{p:g} %": "FP32" for p in PERCENTILES}, BER_TARGETS[data.region])


def compare_variants(sim, cfg, data, x_cal, work: Path, n_slots: int, out: Path) -> None:
    models = ROOT / "models"
    fp32 = models / f"{STANDARD}.onnx"
    eight_bit = work / "eight_bit.onnx"
    quantize(fp32, eight_bit, x_cal, reduce_range=False)
    min_max = work / "min_max.onnx"
    quantize(fp32, min_max, x_cal, percentile=None)
    whole = work / "whole.onnx"
    export_onnx(load_model(models / f"{STANDARD}.pt"), whole, whole_model=True)
    naive = work / "naive.onnx"
    quantize(whole, naive, x_cal, percentile=None, per_channel=False, reduce_range=False)

    estimators = {
        "LS + linear": InterpolationEstimator(sim.pattern, "linear"),
        "standard FP32": OnnxEstimator(fp32, sim.pattern),
        "standard int8": OnnxEstimator(models / f"{STANDARD}_int8.onnx", sim.pattern),
        "standard int8, 8-bit": OnnxEstimator(eight_bit, sim.pattern),
        "standard int8, min/max": OnnxEstimator(min_max, sim.pattern),
        "standard int8, naive": WholeModelEstimator(naive, sim.pattern),
        "small FP32": OnnxEstimator(models / f"{SMALL}.onnx", sim.pattern),
        "small int8": OnnxEstimator(models / f"{SMALL}_int8.onnx", sim.pattern),
    }
    reference = {
        "standard int8": "standard FP32",
        "standard int8, 8-bit": "standard FP32",
        "standard int8, min/max": "standard FP32",
        "standard int8, naive": "standard FP32",
        "small int8": "small FP32",
    }
    print("One int8 step at the output, and the operators that really run:")
    for label, path in (
        ("standard int8", models / f"{STANDARD}_int8.onnx"),
        ("standard int8, 8-bit", eight_bit),
        ("standard int8, min/max", min_max),
        ("standard int8, naive", naive),
        ("small int8", models / f"{SMALL}_int8.onnx"),
    ):
        print(f"{label:26s}{output_step(path):8.4f}  {ops_that_run(path)}")

    rows = []
    for region_name, target in BER_TARGETS.items():
        rng = np.random.default_rng(data.test_seeds[region_name])
        region_rows = evaluate(
            sim, estimators, cfg.regions[region_name], cfg.evaluation.snr_db, n_slots,
            cfg.evaluation.batch_size, rng,
        )
        rows += [{"region": region_name, **row} for row in region_rows]
        print(f"\nRegion {region_name}, test slots, {n_slots} slots per SNR")
        print_table(region_rows, reference, target)
    write_csv(rows, out / "quantization.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--nn-config", type=Path, default=ROOT / "configs" / "nn.yaml")
    parser.add_argument("--part", choices=["all", "percentile", "variants"], default="all")
    parser.add_argument("--n-slots", type=int, help="override data.test_slots")
    parser.add_argument("--percentile-slots", type=int, default=2000, help="slots per SNR")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = load_nn_config(args.nn_config).data
    sim = LinkSimulator(cfg)
    x_cal = calibration_input(sim, cfg, data)
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        if args.part in ("all", "percentile"):
            compare_percentiles(sim, cfg, data, x_cal, work, args.percentile_slots, args.out)
        if args.part in ("all", "variants"):
            n_slots = args.n_slots or data.test_slots
            compare_variants(sim, cfg, data, x_cal, work, n_slots, args.out)
    print(f"\n-> {args.out / 'quantization.csv'}, quantization_percentile.csv")


if __name__ == "__main__":
    main()
