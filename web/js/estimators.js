// The receiver of the demo: LS + linear as in src/ofdm/estimators.py, the network input as in
// src/nn/dataset.py, the CNN on top of it, then equalization, hard decisions and error measures.

import { makeComplex } from "./ofdm.js";
import { INPUT_SIZE, OUTPUT_SIZE, prepare, runCorrection } from "./network.js";

// Piecewise linear interpolation from the sorted positions in known onto 0, 1, ..., length - 1.
// Outside the known points the first or last segment continues, as in Python.
function linearWeights(known, length) {
  const lower = new Int32Array(length);
  const weight = new Float64Array(length);
  for (let x = 0; x < length; x++) {
    let upper = 1;
    while (upper < known.length - 1 && known[upper] <= x) {
      upper++;
    }
    lower[x] = upper - 1;
    weight[x] = (x - known[upper - 1]) / (known[upper] - known[upper - 1]);
  }
  return { lower, weight };
}

// model is the parsed JSON of an int8 network from scripts/export_web_data.py.
export function makeReceiver(link, model) {
  const { n_symbols: nSymbols, n_subcarriers: nSubcarriers, pilots } = link.config;
  const bySymbol = new Map();
  pilots.symbol.forEach((symbol, i) => {
    bySymbol.set(symbol, [...(bySymbol.get(symbol) ?? []), pilots.subcarrier[i]]);
  });
  const rows = [...bySymbol].map(([symbol, subcarriers]) => ({
    symbol,
    subcarriers,
    weights: linearWeights(subcarriers, nSubcarriers),
  }));
  return {
    link,
    rows,
    time: linearWeights(
      rows.map((row) => row.symbol),
      nSymbols,
    ),
    network: prepare(model),
    input: new Float32Array(INPUT_SIZE),
    correction: new Float32Array(OUTPUT_SIZE),
  };
}

// Least squares at the pilots, then linear interpolation: first over frequency, then over time.
export function lsLinear(receiver, rxGrid) {
  const { link, rows, time } = receiver;
  const { n_symbols: nSymbols, n_subcarriers: nSubcarriers } = link.config;
  const { pilotGrid } = link;
  const overFrequency = makeComplex(rows.length * nSubcarriers);
  rows.forEach(({ symbol, subcarriers, weights }, t) => {
    const lsRe = new Float64Array(subcarriers.length);
    const lsIm = new Float64Array(subcarriers.length);
    subcarriers.forEach((k, i) => {
      const element = symbol * nSubcarriers + k;
      const xr = pilotGrid.re[element];
      const xi = pilotGrid.im[element];
      const power = xr * xr + xi * xi;
      lsRe[i] = (rxGrid.re[element] * xr + rxGrid.im[element] * xi) / power;
      lsIm[i] = (rxGrid.im[element] * xr - rxGrid.re[element] * xi) / power;
    });
    for (let k = 0; k < nSubcarriers; k++) {
      const j = weights.lower[k];
      const w = weights.weight[k];
      overFrequency.re[t * nSubcarriers + k] = (1 - w) * lsRe[j] + w * lsRe[j + 1];
      overFrequency.im[t * nSubcarriers + k] = (1 - w) * lsIm[j] + w * lsIm[j + 1];
    }
  });
  const estimate = makeComplex(nSymbols * nSubcarriers);
  for (let s = 0; s < nSymbols; s++) {
    const before = time.lower[s] * nSubcarriers;
    const after = before + nSubcarriers;
    const w = time.weight[s];
    for (let k = 0; k < nSubcarriers; k++) {
      const element = s * nSubcarriers + k;
      estimate.re[element] = (1 - w) * overFrequency.re[before + k] + w * overFrequency.re[after + k];
      estimate.im[element] = (1 - w) * overFrequency.im[before + k] + w * overFrequency.im[after + k];
    }
  }
  return estimate;
}

// Channels: real part, imaginary part, and the SNR as a constant plane. Float32Array rounds like
// astype(np.float32) in Python.
export function toInput(estimate, snrDb, snrScaleDb, input) {
  const n = estimate.re.length;
  input.set(estimate.re, 0);
  input.set(estimate.im, n);
  input.fill(snrDb / snrScaleDb, 2 * n, 3 * n);
  return input;
}

// The network only computes a correction, which is added to its own input, in float32 as in Python.
export function cnnEstimate(receiver, lsEstimate, snrDb) {
  const { input, correction, network, link } = receiver;
  toInput(lsEstimate, snrDb, link.config.snr_scale_db, input);
  runCorrection(network, input, correction);
  const n = lsEstimate.re.length;
  const estimate = makeComplex(n);
  for (let i = 0; i < n; i++) {
    estimate.re[i] = Math.fround(input[i] + correction[i]);
    estimate.im[i] = Math.fround(input[n + i] + correction[n + i]);
  }
  return estimate;
}

// One-tap zero-forcing equalization on the data elements: Y / H.
export function equalize(link, rxGrid, estimate) {
  const { dataElements } = link;
  const equalized = makeComplex(dataElements.length);
  dataElements.forEach((element, i) => {
    const hr = estimate.re[element];
    const hi = estimate.im[element];
    const power = hr * hr + hi * hi;
    equalized.re[i] = (rxGrid.re[element] * hr + rxGrid.im[element] * hi) / power;
    equalized.im[i] = (rxGrid.im[element] * hr - rxGrid.re[element] * hi) / power;
  });
  return equalized;
}

// Hard QPSK decisions: the sign of the real part gives bit 0, the sign of the imaginary part bit 1.
export function countBitErrors(equalized, bits) {
  let errors = 0;
  for (let i = 0; i < equalized.re.length; i++) {
    errors += (equalized.re[i] > 0 ? 1 : 0) === bits[2 * i] ? 0 : 1;
    errors += (equalized.im[i] > 0 ? 1 : 0) === bits[2 * i + 1] ? 0 : 1;
  }
  return errors;
}

export function squaredError(estimate, truth) {
  let sum = 0;
  for (let i = 0; i < truth.re.length; i++) {
    sum += (estimate.re[i] - truth.re[i]) ** 2 + (estimate.im[i] - truth.im[i]) ** 2;
  }
  return sum;
}

export function energy(grid) {
  let sum = 0;
  for (let i = 0; i < grid.re.length; i++) {
    sum += grid.re[i] ** 2 + grid.im[i] ** 2;
  }
  return sum;
}

// All three estimators on one simulated slot. slot comes from simulateSlot in link.js.
export function receiveSlot(receiver, slot, snrDb) {
  const { link } = receiver;
  const ls = lsLinear(receiver, slot.rxGrid);
  const estimates = { ls, cnn: cnnEstimate(receiver, ls, snrDb), perfect: slot.hTrue };
  const results = {};
  for (const [name, estimate] of Object.entries(estimates)) {
    const equalized = equalize(link, slot.rxGrid, estimate);
    results[name] = {
      estimate,
      equalized,
      bitErrors: countBitErrors(equalized, slot.bits),
      squaredError: squaredError(estimate, slot.hTrue),
    };
  }
  return { ...results, channelEnergy: energy(slot.hTrue), nBits: slot.bits.length };
}
