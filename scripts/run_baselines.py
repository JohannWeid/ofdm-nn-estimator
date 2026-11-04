"""NMSE and BER over SNR for the classical baselines in one channel region.

Example:
    python scripts/run_baselines.py --region A
    python scripts/run_baselines.py --region B --n-slots 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ofdm import LinkSimulator, baseline_estimators, evaluate, load_config
from ofdm.plotting import plot_curves, print_table, write_csv

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--region", default="A", help="region name from the config")
    parser.add_argument("--n-slots", type=int, help="override evaluation.n_slots")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    region = cfg.regions[args.region]
    n_slots = args.n_slots or cfg.evaluation.n_slots
    sim = LinkSimulator(cfg)
    estimators = baseline_estimators(sim.pattern, cfg.ofdm)
    rng = np.random.default_rng(cfg.seed)

    t0 = time.perf_counter()
    rows = evaluate(
        sim, estimators, region, cfg.evaluation.snr_db, n_slots, cfg.evaluation.batch_size, rng
    )
    elapsed = time.perf_counter() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"baselines_{args.region}"
    write_csv(rows, args.out / f"{stem}.csv")
    title = (
        f"Region {args.region}: delay spread {region.delay_spread_us[0]}-"
        f"{region.delay_spread_us[1]} us, Doppler {region.doppler_hz[0]:.0f}-"
        f"{region.doppler_hz[1]:.0f} Hz, {cfg.ofdm.modulation.upper()}, {n_slots} slots/SNR"
    )
    plot_curves(rows, title, args.out / f"{stem}.png")

    print(f"{title}  ({elapsed:.1f} s)")
    print_table(rows, cfg.evaluation.snr_db)
    print(f"-> {args.out / stem}.csv, .png")


if __name__ == "__main__":
    main()
