"""What does each building block contribute? Model variants on the same independent test slots.

Example:
    python scripts/compare_ablations.py
    python scripts/compare_ablations.py --n-slots 500
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from nn.config import NNConfig, load_nn_config
from nn.estimator import NeuralEstimator
from nn.model import count_parameters, load_model, macs_per_slot
from ofdm import LinkSimulator, evaluate, load_config
from ofdm.estimators import InterpolationEstimator
from ofdm.plotting import write_csv

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "configs" / "experiments"
VARIANTS = {
    "standard": ROOT / "configs" / "nn.yaml",
    "other seed": EXPERIMENTS / "seed_1.yaml",
    "no SNR plane": EXPERIMENTS / "no_snr.yaml",
    "small model": EXPERIMENTS / "small.yaml",
    "30 % data": EXPERIMENTS / "data_30pct.yaml",
    "10 % data": EXPERIMENTS / "data_10pct.yaml",
}


def training_summary(nn_cfg: NNConfig, results: Path) -> dict[str, float]:
    """Best epoch and gradient steps from the training log; NaN if there is no log."""
    log = results / f"training_{nn_cfg.name}.csv"
    if not log.exists():
        return {"best_epoch": float("nan"), "gradient_steps": float("nan")}
    with open(log, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    best = min(rows, key=lambda r: float(r["val_loss"]))
    steps_per_epoch = np.ceil(nn_cfg.data.train.n_slots / nn_cfg.training.batch_size)
    return {"best_epoch": int(best["epoch"]), "gradient_steps": int(len(rows) * steps_per_epoch)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--n-slots", type=int, help="override data.test_slots")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim = LinkSimulator(cfg)
    estimators = {"LS + linear": InterpolationEstimator(sim.pattern, "linear")}
    size_rows = []
    base_data = load_nn_config(VARIANTS["standard"]).data
    for label, path in VARIANTS.items():
        nn_cfg = load_nn_config(path)
        model = load_model(args.models / f"cnn_{nn_cfg.name}.pt")
        estimators[label] = NeuralEstimator(model, sim.pattern)
        size_rows.append(
            {
                "variant": label,
                "parameters": count_parameters(model),
                "macs_per_slot": macs_per_slot(model),
                "train_slots": nn_cfg.data.train.n_slots,
                **training_summary(nn_cfg, ROOT / "results"),
            }
        )

    n_slots = args.n_slots or base_data.test_slots
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for region_name in ("A", "B"):
        rng = np.random.default_rng(base_data.test_seeds[region_name])
        region = cfg.regions[region_name]
        batch = cfg.evaluation.batch_size
        for row in evaluate(sim, estimators, region, cfg.evaluation.snr_db, n_slots, batch, rng):
            rows.append({"region": region_name, **row})
    write_csv(rows, args.out / "ablations.csv")
    write_csv(size_rows, args.out / "ablations_size.csv")

    header = ["parameters", "M MACs", "slots", "best ep.", "steps"]
    print(f"{'variant':14s}" + "".join(f"{h:>10s}" for h in header))
    for s in size_rows:
        print(f"{s['variant']:14s}{s['parameters']:10d}{s['macs_per_slot'] / 1e6:10.2f}"
              f"{s['train_slots']:10d}{s['best_epoch']:10.0f}{s['gradient_steps']:10.0f}")
    snrs = cfg.evaluation.snr_db
    for region_name in ("A", "B"):
        print(f"\nRegion {region_name}: NMSE gain over LS + linear (dB)")
        print(f"{'SNR':14s}" + "".join(f"{s:7.0f}" for s in snrs))
        in_region = [r for r in rows if r["region"] == region_name]
        nmse = {(r["estimator"], r["snr_db"]): r["nmse_db"] for r in in_region}
        for label in VARIANTS:
            gains = [nmse[("LS + linear", float(s))] - nmse[(label, float(s))] for s in snrs]
            print(f"{label:14s}" + "".join(f"{g:7.2f}" for g in gains))
    print(f"\n-> {args.out / 'ablations.csv'}, ablations_size.csv")


if __name__ == "__main__":
    main()
