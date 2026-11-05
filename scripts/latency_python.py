"""Latency in Python on this machine: PyTorch and ONNX Runtime, one thread, one slot at a time.

The numbers that count are the ones from cpp/bench/bench_latency.cpp. This script is for comparisons
that C++ cannot show: PyTorch against ONNX Runtime, one operating system against another, and how
the time is spread over the layers. Without PyTorch it measures ONNX Runtime only.

Writes results/latency_python_<system>.csv and results/latency_layers_<system>.csv.

Example:
    python scripts/latency_python.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("cnn_A_log_nmse", "cnn_A_small")
N_REAL_SLOTS = 4  # the fifth slot in input.bin is the artificial one with doubled values
WARMUP_RUNS = 200
PROFILE_RUNS = 500


def write_rows(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def one_thread_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options


def onnx_runner(path: Path):
    session = ort.InferenceSession(
        str(path), one_thread_options(), providers=["CPUExecutionProvider"]
    )
    return lambda slot: session.run(None, {"input": slot})


def pytorch_runner(path: Path):
    import torch

    from nn.model import load_model

    torch.set_num_threads(1)
    body = load_model(path).body.eval()

    def run(slot: np.ndarray) -> None:
        with torch.inference_mode():
            body(torch.from_numpy(slot))

    return run


def measure(run, slots: list[np.ndarray], n_runs: int) -> dict[str, float]:
    for i in range(WARMUP_RUNS):
        run(slots[i % len(slots)])
    ms = np.empty(n_runs)
    for i in range(n_runs):
        start = time.perf_counter()
        run(slots[i % len(slots)])
        ms[i] = (time.perf_counter() - start) * 1e3
    return {
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "max_ms": float(ms.max()),
    }


def layer_times(path: Path, slots: list[np.ndarray]) -> list[tuple[str, float]]:
    """Median kernel time per executed node in microseconds, from the ONNX Runtime profiler."""
    with tempfile.TemporaryDirectory() as tmp:
        options = one_thread_options()
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(tmp) / "profile")
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        for i in range(PROFILE_RUNS):
            session.run(None, {"input": slots[i % len(slots)]})
        with open(session.end_profiling(), encoding="utf-8") as f:
            events = json.load(f)
    nodes: dict[tuple[str, str], list[float]] = {}
    for event in events:
        if event.get("cat") == "Node" and event["name"].endswith("_kernel_time"):
            nodes.setdefault((event["args"]["op_name"], event["name"]), []).append(event["dur"])
    return [(op, float(np.median(durations))) for (op, _), durations in nodes.items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=5000, help="runs per repeat")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    models = ROOT / "models"
    x = np.fromfile(ROOT / "cpp" / "testdata" / "input.bin", dtype="<f4").reshape(-1, 3, 14, 72)
    slots = [x[i : i + 1] for i in range(N_REAL_SLOTS)]

    has_pytorch = importlib.util.find_spec("torch") is not None
    runners = {}
    for name in MODELS:
        if has_pytorch:
            runners[("PyTorch", name)] = pytorch_runner(models / f"{name}.pt")
        runners[("ONNX Runtime", name)] = onnx_runner(models / f"{name}.onnx")
        runners[("ONNX Runtime", f"{name}_int8")] = onnx_runner(models / f"{name}_int8.onnx")

    system = platform.system().lower()
    rows = []
    for repeat in range(args.repeats):  # the models take turns, like in the C++ benchmark
        for (runtime, name), run in runners.items():
            latency = measure(run, slots, args.runs)
            rows.append({"system": system, "runtime": runtime, "model": name, "repeat": repeat,
                         "runs": args.runs, **latency})
            print(f"{runtime:13s}{name:22s}{repeat:3d}  median {latency['median_ms']:.3f} ms, "
                  f"p99 {latency['p99_ms']:.3f} ms, max {latency['max_ms']:.2f} ms", flush=True)
    write_rows(rows, args.out / f"latency_python_{system}.csv")

    layer_rows = []
    for name in (n for model in MODELS for n in (model, f"{model}_int8")):
        times = layer_times(models / f"{name}.onnx", slots)
        print(f"{name:22s}" + "  ".join(f"{op} {us:.0f}" for op, us in times))
        layer_rows += [{"system": system, "model": name, "node": i, "op": op, "median_us": us}
                       for i, (op, us) in enumerate(times)]
    write_rows(layer_rows, args.out / f"latency_layers_{system}.csv")
    print(f"-> {args.out / f'latency_python_{system}.csv'}, latency_layers_{system}.csv")


if __name__ == "__main__":
    main()
