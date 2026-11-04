"""Where does the CNN break? NMSE difference to the baselines over delay spread and Doppler.

Every estimator sees the same slots in a cell, so the differences are more precise than the
slot count alone suggests.

Example:
    python scripts/failure_map.py
    python scripts/failure_map.py --n-slots 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from nn.config import load_nn_config
from nn.estimator import NeuralEstimator
from nn.model import load_model
from ofdm import LinkSimulator, baseline_estimators, load_config
from ofdm.config import Region
from ofdm.link import evaluate_grid
from ofdm.plotting import INK, MUTED, SURFACE, write_csv

ROOT = Path(__file__).resolve().parents[1]

# Diverging: blue = CNN better than the reference, gray = equal, red = CNN worse.
CMAP = LinearSegmentedColormap.from_list("cnn_diff", ["#184f95", "#f0efec", "#e34948"])
SCALE_DB = 8.0


def difference_grid(
    rows: list[dict], reference: str, snr: float, delays: list[float], dopplers: list[float]
) -> np.ndarray:
    """NMSE(CNN) - NMSE(reference) in dB with shape (len(dopplers), len(delays))."""
    nmse = {
        (r["delay_spread_us"], r["doppler_hz"], r["estimator"]): r["nmse_db"]
        for r in rows
        if r["snr_db"] == snr
    }
    return np.array(
        [[nmse[(d, f, "cnn")] - nmse[(d, f, reference)] for d in delays] for f in dopplers]
    )


def draw_region(ax, name: str, region: Region, delays: list[float], dopplers: list[float]) -> None:
    """Frame the cells whose center lies in the region, along cell edges to keep labels clear."""
    inside_x = [d for d in delays if region.delay_spread_us[0] <= d <= region.delay_spread_us[1]]
    inside_y = [f for f in dopplers if region.doppler_hz[0] <= f <= region.doppler_hz[1]]
    x0, x1 = inside_x[0] - 0.1, inside_x[-1] + 0.1
    y0, y1 = inside_y[0] - 50.0, inside_y[-1] + 50.0
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=INK, linewidth=2.0))
    ax.text(x0 + 0.015, y1 + 6, name, color=INK, fontsize=9, fontweight="bold", va="bottom")


def plot_map(
    rows: list[dict],
    delays: list[float],
    dopplers: list[float],
    snrs: list[float],
    regions: dict[str, Region],
    path: Path,
) -> None:
    references = [("ls_linear", "CNN minus LS + linear"), ("lmmse", "CNN minus LMMSE")]
    fig = Figure(figsize=(14, 8), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(len(references), len(snrs), sharex=True, sharey=True, squeeze=False)
    norm = Normalize(-SCALE_DB, SCALE_DB)
    x_edges = np.append(np.array(delays) - 0.1, delays[-1] + 0.1)
    y_edges = np.append(np.array(dopplers) - 50.0, dopplers[-1] + 50.0)

    for i, (reference, label) in enumerate(references):
        for j, snr in enumerate(snrs):
            ax = axes[i, j]
            diff = difference_grid(rows, reference, snr, delays, dopplers)
            mesh = ax.pcolormesh(
                x_edges, y_edges, diff, cmap=CMAP, norm=norm, edgecolors=SURFACE, linewidth=1.0
            )
            for row, doppler in enumerate(dopplers):
                for col, delay in enumerate(delays):
                    value = diff[row, col]
                    color = "#ffffff" if abs(value) > 0.6 * SCALE_DB else INK
                    ax.text(delay, doppler, f"{value:+.1f}", ha="center", va="center",
                            fontsize=7.5, color=color)
            for name, region in regions.items():
                draw_region(ax, name, region, delays, dopplers)
            ax.set_title(f"{label}, {snr:.0f} dB SNR", color=INK, fontsize=10)
            ax.tick_params(colors=MUTED, labelcolor=INK)
            ax.set_xticks(delays)
            ax.set_yticks(dopplers)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == len(references) - 1:
                ax.set_xlabel("Delay spread (us)", color=INK)
            if j == 0:
                ax.set_ylabel("Doppler (Hz)", color=INK)
    bar = fig.colorbar(mesh, ax=axes, shrink=0.7, label="NMSE difference (dB): blue = CNN better")
    bar.ax.tick_params(colors=MUTED, labelcolor=INK)
    bar.outline.set_visible(False)
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--nn-config", type=Path, default=ROOT / "configs" / "nn.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "cnn_A.pt")
    parser.add_argument("--n-slots", type=int, help="override failure_map.n_slots")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    map_cfg = load_nn_config(args.nn_config).failure_map
    sim = LinkSimulator(cfg)
    baselines = baseline_estimators(sim.pattern, cfg.ofdm)
    estimators = {
        "ls_linear": baselines["ls_linear"],
        "lmmse": baselines["lmmse"],
        "cnn": NeuralEstimator(load_model(args.model), sim.pattern),
    }
    delays = [float(d) for d in map_cfg.delay_spread_us]
    dopplers = [float(f) for f in map_cfg.doppler_hz]
    snrs = [float(s) for s in map_cfg.snr_db]

    t0 = time.perf_counter()
    rows = evaluate_grid(
        sim, estimators, delays, dopplers, snrs, args.n_slots or map_cfg.n_slots,
        cfg.evaluation.batch_size, np.random.default_rng(map_cfg.seed),
    )
    print(f"{len(delays)} x {len(dopplers)} cells, {len(snrs)} SNRs "
          f"in {(time.perf_counter() - t0) / 60:.1f} min")

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "failure_map.csv")
    plot_map(rows, delays, dopplers, snrs, cfg.regions, args.out / "failure_map.png")

    for snr in snrs:
        diff = difference_grid(rows, "ls_linear", snr, delays, dopplers)
        print(f"\nCNN minus LS + linear at {snr:.0f} dB SNR (dB, negative = CNN better); "
              f"worse in {np.sum(diff > 0)} of {diff.size} cells")
        print("Doppler \\ delay " + " ".join(f"{d:5.1f}" for d in delays))
        for f, line in zip(dopplers, diff, strict=True):
            print(f"{f:15.0f} " + " ".join(f"{v:+5.1f}" for v in line))
    print(f"\n-> {args.out / 'failure_map'}.csv, .png")


if __name__ == "__main__":
    main()
