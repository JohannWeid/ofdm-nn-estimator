"""Write the data for the browser demo as JSON.

    web/model/<model>.json     the int8 network from models/<model>.onnx
    web/model/link.json        OFDM, pilot, channel and region parameters from the configuration
    web/tests/vectors/simulator_slot.json    one slot from the Python simulator, for the tests

Run this again after scripts/export_onnx.py has changed a model.

Example:
    python scripts/export_web_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nn.web_data import int8_layers, link_data, simulator_vector
from ofdm import load_config

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("cnn_A_log_nmse_int8", "cnn_A_small_int8")


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"{path.name}: {path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path, default=ROOT / "web")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for name in MODELS:
        write_json(int8_layers(args.models / f"{name}.onnx"), args.out / "model" / f"{name}.json")
    write_json(link_data(cfg), args.out / "model" / "link.json")
    vector = simulator_vector(cfg, args.models / f"{MODELS[0]}.onnx")
    write_json(vector, args.out / "tests" / "vectors" / "simulator_slot.json")


if __name__ == "__main__":
    main()
