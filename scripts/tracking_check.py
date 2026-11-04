"""Does an estimator smooth real channel variation away? Tests the failure map's explanation.

For each channel condition this prints which share of the true channel change the estimators
reproduce, over 3 OFDM symbols (time) and over 6 subcarriers (frequency).

Example:
    python scripts/tracking_check.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nn.estimator import NeuralEstimator
from nn.model import load_model
from ofdm import LinkSimulator, baseline_estimators, fixed_profiles, load_config
from ofdm.metrics import tracked_variation
from ofdm.plotting import write_csv

ROOT = Path(__file__).resolve().parents[1]
# name, delay spread in us, Doppler in Hz
CONDITIONS = [
    ("region A", 0.3, 50.0),
    ("high Doppler", 0.3, 600.0),
    ("large delay spread", 1.3, 0.0),
    ("region B", 1.1, 450.0),
]
DIRECTIONS = [("time", 1, 3), ("frequency", 2, 6)]  # name, axis, lag in cells
SNRS_DB = [10.0, 20.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "cnn_A_log_nmse.pt")
    parser.add_argument("--n-slots", type=int, default=800, help="slots per condition and SNR")
    parser.add_argument("--seed", type=int, default=105)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim = LinkSimulator(cfg)
    baselines = baseline_estimators(sim.pattern, cfg.ofdm)
    estimators = {
        "ls_linear": baselines["ls_linear"],
        "cnn": NeuralEstimator(load_model(args.model), sim.pattern),
        "lmmse": baselines["lmmse"],
    }
    rng = np.random.default_rng(args.seed)
    batch = cfg.evaluation.batch_size

    rows = []
    for snr in SNRS_DB:
        print(f"\nSNR {snr:.0f} dB: share of the true channel change that arrives")
        print(f"{'condition':22s}{'direction':>11s}" + "".join(f"{n:>11s}" for n in estimators))
        for name, tau_us, doppler in CONDITIONS:
            h_true, h_hat = [], {n: [] for n in estimators}
            for start in range(0, args.n_slots, batch):
                b = min(batch, args.n_slots - start)
                slots = sim.simulate(b, snr, rng, fixed_profiles(tau_us * 1e-6, doppler, b))
                h_true.append(slots.h_true)
                for n, estimator in estimators.items():
                    h_hat[n].append(estimator(slots))
            truth = np.concatenate(h_true)
            for direction, axis, lag in DIRECTIONS:
                shares = {
                    n: tracked_variation(np.concatenate(v), truth, axis, lag)
                    for n, v in h_hat.items()
                }
                if np.isnan(shares["cnn"]):
                    continue  # no Doppler: the channel does not change over time
                print(f"{name:22s}{direction:>11s}"
                      + "".join(f"{s:11.2f}" for s in shares.values()))
                rows.append(
                    {"snr_db": snr, "condition": name, "delay_spread_us": tau_us,
                     "doppler_hz": doppler, "direction": direction, **shares}
                )

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "tracking.csv")
    print(f"\n-> {args.out / 'tracking.csv'}")


if __name__ == "__main__":
    main()
