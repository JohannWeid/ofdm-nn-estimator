"""Training loop: Adam, MSE loss, early stopping on the validation loss, CSV log."""

from __future__ import annotations

import copy
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import TrainConfig

LOG_FIELDS = ["epoch", "train_loss", "val_loss", "val_nmse_db", "seconds"]


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    best_val_nmse_db: float
    history: list[dict[str, float]]


def nmse_db(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """NMSE over all slots in dB; Re and Im are the two channels."""
    err = torch.sum((prediction - target) ** 2)
    return float(10 * torch.log10(err / torch.sum(target**2)))


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i : i + batch_size]) for i in range(0, len(x), batch_size)])


def train(
    model: torch.nn.Module,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
    cfg: TrainConfig,
    seed: int,
    log_path: Path | None = None,
    verbose: bool = True,
) -> TrainResult:
    """Train in place and leave the model at its best validation epoch."""
    torch.manual_seed(seed)
    x_train, y_train = (torch.from_numpy(a) for a in train_data)
    x_val, y_val = (torch.from_numpy(a) for a in val_data)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),  # reproducible batch order
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.MSELoss()

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, LOG_FIELDS).writeheader()

    best_loss, best_epoch, best_state = float("inf"), 0, copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            count += len(xb)

        val_pred = predict(model, x_val)
        row = {
            "epoch": epoch,
            "train_loss": total / count,
            "val_loss": loss_fn(val_pred, y_val).item(),
            "val_nmse_db": nmse_db(val_pred, y_val),
            "seconds": time.perf_counter() - t0,
        }
        history.append(row)
        if log_path is not None:
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, LOG_FIELDS).writerow(row)
        if verbose:
            print(
                f"epoch {epoch:3d}  train {row['train_loss']:.5f}  val {row['val_loss']:.5f}"
                f"  val NMSE {row['val_nmse_db']:6.2f} dB  ({row['seconds']:.1f} s)",
                flush=True,
            )

        if row["val_loss"] < best_loss:
            best_loss, best_epoch = row["val_loss"], epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            if verbose:
                print(f"early stopping: no improvement since epoch {best_epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    best_nmse = history[best_epoch - 1]["val_nmse_db"] if best_epoch else float("nan")
    return TrainResult(best_epoch, best_loss, best_nmse, history)
