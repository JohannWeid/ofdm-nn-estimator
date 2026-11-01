"""NMSE and BER over SNR for the classical baselines in one channel region.

Example:
    python scripts/run_baselines.py --region A
    python scripts/run_baselines.py --region B --n-slots 200
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ofdm import LinkSimulator, baseline_estimators, evaluate, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Chart styling: three categorical slots (validated light palette) + neutral reference line.
STYLE = {
    "ls_linear": {"label": "LS + linear", "color": "#2a78d6", "marker": "o"},
    "ls_spline": {"label": "LS + spline", "color": "#eb6834", "marker": "s"},
    "lmmse": {"label": "LMMSE (known statistics)", "color": "#1baf7a", "marker": "^"},
    "perfect": {"label": "Perfect CSI", "color": "#52514e", "marker": "", "linestyle": "--"},
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], title: str, path: Path) -> None:
    fig, (ax_nmse, ax_ber) = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    for name, style in STYLE.items():
        sel = sorted((r for r in rows if r["estimator"] == name), key=lambda r: r["snr_db"])
        if not sel:
            continue
        snr = [r["snr_db"] for r in sel]
        kw = {
            "label": style["label"],
            "color": style["color"],
            "marker": style["marker"],
            "linestyle": style.get("linestyle", "-"),
            "linewidth": 1.5,
            "markersize": 6,
        }
        if name != "perfect":  # NMSE of the genie is -inf dB
            ax_nmse.plot(snr, [r["nmse_db"] for r in sel], **kw)
        ber = [r["ber"] if r["ber"] > 0 else np.nan for r in sel]
        ax_ber.semilogy(snr, ber, **kw)

    ax_nmse.set_ylabel("NMSE (dB)")
    ax_ber.set_ylabel("BER")
    for ax in (ax_nmse, ax_ber):
        ax.set_facecolor(SURFACE)
        ax.set_xlabel("SNR per resource element (dB)")
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelcolor=INK)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)
    ax_ber.legend(frameon=False, labelcolor=INK)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    plot(rows, title, args.out / f"{stem}.png")

    print(f"{title}  ({elapsed:.1f} s)")
    print(f"{'SNR':>5}  " + "  ".join(f"{STYLE[n]['label']:>26}" for n in estimators))
    for snr in cfg.evaluation.snr_db:
        cells = []
        for name in estimators:
            r = next(r for r in rows if r["snr_db"] == snr and r["estimator"] == name)
            cells.append(f"{r['nmse_db']:7.1f} dB / BER {r['ber']:.2e}")
        print(f"{snr:5.0f}  " + "  ".join(f"{c:>26}" for c in cells))
    print(f"-> {args.out / stem}.csv, .png")


if __name__ == "__main__":
    main()
