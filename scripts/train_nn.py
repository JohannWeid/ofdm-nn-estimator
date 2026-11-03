"""Train the residual CNN on slots from one channel region.

Examples:
    python scripts/train_nn.py                  # full training -> models/cnn_A.pt
    python scripts/train_nn.py --overfit 256    # sanity check: can it memorize 256 slots?
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from nn.config import load_nn_config  # noqa: E402
from nn.dataset import make_dataset  # noqa: E402
from nn.model import build_model, count_parameters, macs_per_slot, save_model  # noqa: E402
from nn.train import nmse_db, train  # noqa: E402
from ofdm import LinkSimulator, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def plot_history(history: list[dict], input_nmse_db: float, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    epochs = [r["epoch"] for r in history]
    ax.plot(epochs, [r["val_nmse_db"] for r in history], color="#2a78d6", linewidth=1.5,
            label="CNN, validation set")
    ax.axhline(input_nmse_db, color="#52514e", linestyle="--", linewidth=1.5,
               label="LS + linear (network input)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NMSE (dB)")
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.legend(frameon=False, labelcolor=INK)
    ax.set_title(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--nn-config", type=Path, default=ROOT / "configs" / "nn.yaml")
    parser.add_argument("--overfit", type=int, metavar="N",
                        help="train and validate on the same N slots, save nothing")
    parser.add_argument("--epochs", type=int, help="override training.max_epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    nn_cfg = load_nn_config(args.nn_config)
    data_cfg, train_cfg = nn_cfg.data, nn_cfg.training
    sim = LinkSimulator(cfg)
    region = cfg.regions[data_cfg.region]

    t0 = time.perf_counter()
    if args.overfit:
        train_data = make_dataset(sim, region, args.overfit, data_cfg.snr_db, data_cfg.train.seed)
        val_data = train_data
        train_cfg = dataclasses.replace(
            train_cfg, batch_size=32, max_epochs=args.epochs or 300, patience=10**9
        )
    else:
        train_data = make_dataset(
            sim, region, data_cfg.train.n_slots, data_cfg.snr_db, data_cfg.train.seed
        )
        val_data = make_dataset(
            sim, region, data_cfg.val.n_slots, data_cfg.snr_db, data_cfg.val.seed
        )
        if args.epochs:
            train_cfg = dataclasses.replace(train_cfg, max_epochs=args.epochs)
    x_val, y_val = (torch.from_numpy(a) for a in val_data)
    input_nmse = nmse_db(x_val[:, :2], y_val)
    print(f"data: {len(train_data[0])} train / {len(val_data[0])} val slots from region "
          f"{data_cfg.region} in {time.perf_counter() - t0:.0f} s")
    print(f"input (LS + linear) NMSE on the validation set: {input_nmse:.2f} dB")

    model = build_model(nn_cfg.model)
    print(f"model: {count_parameters(model)} parameters, "
          f"{macs_per_slot(model) / 1e6:.2f} M MACs per slot")

    stem = f"cnn_{data_cfg.region}"
    log_path = None if args.overfit else ROOT / "results" / f"training_{data_cfg.region}.csv"
    t0 = time.perf_counter()
    result = train(model, train_data, val_data, train_cfg, nn_cfg.seed, log_path,
                   verbose=not args.overfit)
    minutes = (time.perf_counter() - t0) / 60

    if args.overfit:
        for row in result.history[:: max(1, len(result.history) // 6)]:
            print(f"epoch {row['epoch']:4d}  NMSE {row['val_nmse_db']:6.2f} dB")
    print(f"best epoch {result.best_epoch}: NMSE {result.best_val_nmse_db:.2f} dB, "
          f"{input_nmse - result.best_val_nmse_db:.2f} dB better than the input "
          f"({minutes:.1f} min)")
    if args.overfit:
        return

    save_model(model, nn_cfg.model, ROOT / "models" / f"{stem}.pt")
    plot_history(result.history, input_nmse,
                 f"Training on region {data_cfg.region} (SNR {data_cfg.snr_db[0]:.0f}-"
                 f"{data_cfg.snr_db[1]:.0f} dB)",
                 ROOT / "results" / f"training_{data_cfg.region}.png")
    print(f"-> models/{stem}.pt, results/training_{data_cfg.region}.csv, .png")


if __name__ == "__main__":
    main()
