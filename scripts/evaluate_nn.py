"""NMSE and BER over SNR: the trained CNN against the baselines, on independent test data.

Example:
    python scripts/evaluate_nn.py --region A
    python scripts/evaluate_nn.py --region B --n-slots 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from nn.config import load_nn_config
from nn.estimator import NeuralEstimator
from nn.model import load_model
from ofdm import LinkSimulator, baseline_estimators, evaluate, load_config
from ofdm.plotting import plot_curves, print_table, write_csv

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--nn-config", type=Path, default=ROOT / "configs" / "nn.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "cnn_A.pt")
    parser.add_argument("--region", default="A", help="region name from the config")
    parser.add_argument("--n-slots", type=int, help="override data.test_slots")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    nn_cfg = load_nn_config(args.nn_config)
    region = cfg.regions[args.region]
    n_slots = args.n_slots or nn_cfg.data.test_slots
    sim = LinkSimulator(cfg)
    estimators = baseline_estimators(sim.pattern, cfg.ofdm)
    estimators["cnn"] = NeuralEstimator(load_model(args.model), sim.pattern)
    rng = np.random.default_rng(nn_cfg.data.test_seeds[args.region])

    t0 = time.perf_counter()
    rows = evaluate(
        sim, estimators, region, cfg.evaluation.snr_db, n_slots, cfg.evaluation.batch_size, rng
    )
    elapsed = time.perf_counter() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"nn_eval_{args.region}"
    write_csv(rows, args.out / f"{stem}.csv")
    trained_on = nn_cfg.data.region
    title = (
        f"Test region {args.region} (trained on {trained_on}): delay spread "
        f"{region.delay_spread_us[0]}-{region.delay_spread_us[1]} us, Doppler "
        f"{region.doppler_hz[0]:.0f}-{region.doppler_hz[1]:.0f} Hz, {n_slots} slots/SNR"
    )
    plot_curves(rows, title, args.out / f"{stem}.png")

    print(f"{title}  ({elapsed:.1f} s)")
    print_table(rows, cfg.evaluation.snr_db)
    print(f"-> {args.out / stem}.csv, .png")


if __name__ == "__main__":
    main()
