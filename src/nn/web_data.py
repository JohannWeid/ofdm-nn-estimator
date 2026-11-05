"""The data the browser demo needs, as plain numbers: int8 networks and link parameters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from ofdm import Config
from ofdm.ofdm import subcarrier_bins
from ofdm.resource_grid import make_pilot_pattern

SUPPORTED_NODES = {"QuantizeLinear", "DequantizeLinear", "Conv", "Relu"}
SUPPORTED_CONV = {
    "kernel_shape": [3, 3],
    "pads": [1, 1, 1, 1],
    "strides": [1, 1],
    "dilations": [1, 1],
    "group": 1,
}


def int8_layers(onnx_path: str | Path) -> dict:
    """The parameters of a quantized CNN, for an implementation without ONNX Runtime.

    A value q stands for the real value (q - zero) * step. Raises ValueError for anything but a
    chain of 3x3 convolutions with padding 1 in QDQ format, because web/js/network.js implements
    only that.
    """
    graph = onnx.load(onnx_path).graph
    if not {node.op_type for node in graph.node} <= SUPPORTED_NODES:
        raise ValueError(f"unsupported operators in {onnx_path}")
    values = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    producer = {name: node for node in graph.node for name in node.output}
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def quantization(node) -> dict:
        return {"step": float(values[node.input[1]]), "zero": int(values[node.input[2]])}

    def activation_after(conv) -> dict:
        node = consumers[conv.output[0]][0]
        while node.op_type == "Relu":
            node = consumers[node.output[0]][0]
        return quantization(node)

    convolutions = [node for node in graph.node if node.op_type == "Conv"]
    if not convolutions or convolutions[0].input[0] not in producer:
        raise ValueError(f"{onnx_path} is not a quantized model in QDQ format")

    first_input = quantization(producer[convolutions[0].input[0]])
    current = first_input
    layers = []
    for conv in convolutions:
        attributes = {a.name: onnx.helper.get_attribute_value(a) for a in conv.attribute}
        if any(attributes.get(k, v) != v for k, v in SUPPORTED_CONV.items()):
            raise ValueError(f"unsupported convolution in {onnx_path}: {attributes}")
        if quantization(producer[conv.input[0]]) != current:
            raise ValueError(f"{onnx_path} is not a simple chain of convolutions")
        weight_node, bias_node = producer[conv.input[1]], producer[conv.input[2]]
        weights = values[weight_node.input[0]]
        n_out, n_in = weights.shape[:2]
        current = activation_after(conv)
        layers.append(
            {
                "in_channels": int(n_in),
                "out_channels": int(n_out),
                "weights": weights.astype(int).ravel().tolist(),
                "weight_steps": np.broadcast_to(values[weight_node.input[1]], (n_out,))
                .astype(float)
                .tolist(),
                "biases": values[bias_node.input[0]].astype(int).tolist(),
                "output": current,
            }
        )
    return {"input": first_input, "layers": layers}


def link_data(cfg: Config) -> dict:
    """OFDM, pilot, channel and region parameters from the configuration.

    The pilot sequence comes from a NumPy generator that JavaScript cannot reproduce, so its values
    are stored here, in the row-major order of ofdm.estimators.ls_at_pilots.
    """
    pattern = make_pilot_pattern(cfg.ofdm, cfg.pilots)
    symbols, subcarriers = pattern.positions()
    values = pattern.values[pattern.mask]
    return {
        "n_fft": cfg.ofdm.n_fft,
        "n_subcarriers": cfg.ofdm.n_subcarriers,
        "n_symbols": cfg.ofdm.n_symbols,
        "cp_length": cfg.ofdm.cp_length,
        "sample_rate_hz": cfg.ofdm.sample_rate_hz,
        "modulation": cfg.ofdm.modulation,
        "subcarrier_bins": subcarrier_bins(cfg.ofdm).tolist(),
        "pilots": {
            "symbol": symbols.tolist(),
            "subcarrier": subcarriers.tolist(),
            "re": values.real.tolist(),
            "im": values.imag.tolist(),
        },
        "channel": {
            "n_sinusoids": cfg.channel.n_sinusoids,
            "max_taps": cfg.channel.max_taps,
            "pdp_floor_db": cfg.channel.pdp_floor_db,
        },
        "regions": {
            name: {
                "delay_spread_us": list(region.delay_spread_us),
                "doppler_hz": list(region.doppler_hz),
            }
            for name, region in cfg.regions.items()
        },
    }
