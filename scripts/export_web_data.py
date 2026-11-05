"""Write the data for the browser demo to web/model/ as JSON.

    <model>.json    the int8 network from models/<model>.onnx, see nn.web_data.int8_layers
    link.json       OFDM, pilot, channel and region parameters from the configuration

Run this again after scripts/export_onnx.py has changed a model.

Example:
    python scripts/export_web_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nn.web_data import int8_layers, link_data
from ofdm import load_config

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("cnn_A_log_nmse_int8", "cnn_A_small_int8")


def write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"{path.name}: {path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path, default=ROOT / "web" / "model")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        write_json(int8_layers(args.models / f"{name}.onnx"), args.out / f"{name}.json")
    write_json(link_data(load_config(args.config)), args.out / "link.json")


if __name__ == "__main__":
    main()
