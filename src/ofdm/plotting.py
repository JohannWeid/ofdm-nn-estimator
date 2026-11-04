"""Shared output helpers for NMSE/BER results: CSV, console table and figure."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

# Colors follow the estimator, so a curve looks the same in every figure.
STYLE = {
    "ls_linear": {"label": "LS + linear", "color": "#2a78d6", "marker": "o"},
    "ls_spline": {"label": "LS + spline", "color": "#eb6834", "marker": "s"},
    "lmmse": {"label": "LMMSE (known statistics)", "color": "#1baf7a", "marker": "^"},
    "cnn": {"label": "CNN (this work)", "color": "#eda100", "marker": "D"},
    "perfect": {"label": "Perfect CSI", "color": "#52514e", "marker": "", "linestyle": "--"},
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict], snr_db: Sequence[float]) -> None:
    names = [n for n in STYLE if any(r["estimator"] == n for r in rows)]
    print(f"{'SNR':>5}  " + "  ".join(f"{STYLE[n]['label']:>26}" for n in names))
    for snr in snr_db:
        cells = []
        for name in names:
            r = next(r for r in rows if r["snr_db"] == snr and r["estimator"] == name)
            cells.append(f"{r['nmse_db']:7.1f} dB / BER {r['ber']:.2e}")
        print(f"{snr:5.0f}  " + "  ".join(f"{c:>26}" for c in cells))


def plot_curves(rows: list[dict], title: str, path: Path) -> None:
    fig = Figure(figsize=(10, 4), facecolor=SURFACE)
    ax_nmse, ax_ber = fig.subplots(1, 2)
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
