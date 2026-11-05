// The JavaScript simulator against the Python one, see scripts/export_web_data.py: with fixed inputs
// it has to produce the same signals, and its random parts have to have the right statistics.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  addNoise,
  applyChannel,
  drawAngles,
  exponentialPdp,
  frequencyResponse,
  generateTaps,
} from "../js/channel.js";
import { buildGrid, makeLink, simulateSlot } from "../js/link.js";
import { demodulate, fft, makeComplex, modulate } from "../js/ofdm.js";
import { Random } from "../js/random.js";

const WEB = new URL("../", import.meta.url);
const config = JSON.parse(readFileSync(new URL("model/link.json", WEB), "utf8"));
const vector = JSON.parse(readFileSync(new URL("tests/vectors/simulator_slot.json", WEB), "utf8"));
const MAX_DEVIATION = 1e-9;

function largestDifference(a, b) {
  assert.equal(a.re.length, b.re.length);
  let largest = 0;
  for (let i = 0; i < a.re.length; i++) {
    largest = Math.max(largest, Math.abs(a.re[i] - b.re[i]), Math.abs(a.im[i] - b.im[i]));
  }
  return largest;
}

function fixedTaps() {
  const angles = { alpha: Float64Array.from(vector.alpha), phi: Float64Array.from(vector.phi) };
  return generateTaps(Float64Array.from(vector.pdp), vector.doppler_hz, angles, config);
}

// A slot of one OFDM symbol is enough where only the statistics of the channel matter.
const oneSymbol = { ...config, n_symbols: 1 };

// Bessel function J0 from its power series, good enough for the small arguments used here.
function besselJ0(x) {
  let term = 1;
  let sum = 1;
  for (let k = 1; k < 40; k++) {
    term *= -(x * x) / (4 * k * k);
    sum += term;
  }
  return sum;
}

test("the FFT agrees with the definition of the DFT", () => {
  const random = new Random(1);
  const n = 128;
  const input = makeComplex(n);
  for (let i = 0; i < n; i++) {
    input.re[i] = random.normal();
    input.im[i] = random.normal();
  }
  const expected = makeComplex(n);
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      const angle = (-2 * Math.PI * k * i) / n;
      expected.re[k] += (input.re[i] * Math.cos(angle) - input.im[i] * Math.sin(angle)) / Math.sqrt(n);
      expected.im[k] += (input.re[i] * Math.sin(angle) + input.im[i] * Math.cos(angle)) / Math.sqrt(n);
    }
  }
  const output = { re: input.re.slice(), im: input.im.slice() };
  fft(output.re, output.im);
  assert.ok(largestDifference(output, expected) < 1e-11);
  fft(output.re, output.im, true);
  assert.ok(largestDifference(output, input) < 1e-12);
});

test("the FFT rejects sizes that are no power of two", () => {
  assert.throws(() => fft(new Float64Array(100), new Float64Array(100)), /power of two/);
});

test("pilots and QPSK give the same resource grid as Python", () => {
  const link = makeLink(config);
  assert.equal(link.nBits, vector.bits.length);
  assert.ok(largestDifference(buildGrid(link, vector.bits), vector.tx_grid) < MAX_DEVIATION);
});

test("OFDM modulation gives the same time signal as Python", () => {
  const signal = modulate(vector.tx_grid, config);
  assert.ok(largestDifference(signal, vector.tx_signal) < MAX_DEVIATION);
});

test("the power delay profile matches Python", () => {
  for (const example of vector.pdp_examples) {
    const pdp = exponentialPdp(example.delay_spread_us, config);
    assert.equal(pdp.length, example.pdp.length, `delay spread ${example.delay_spread_us}`);
    pdp.forEach((p, l) => assert.ok(Math.abs(p - example.pdp[l]) < 1e-12));
  }
});

test("with the same angles, the channel gives the same received signal as Python", () => {
  const output = applyChannel(vector.tx_signal, fixedTaps());
  assert.ok(largestDifference(output, vector.rx_signal) < MAX_DEVIATION);
});

test("with the same angles, the true channel per resource element matches Python", () => {
  const grid = frequencyResponse(fixedTaps(), config);
  assert.ok(largestDifference(grid, vector.h_true) < MAX_DEVIATION);
});

test("OFDM demodulation gives the same resource grid as Python", () => {
  const signal = {
    re: Float64Array.from(vector.rx_signal.re),
    im: Float64Array.from(vector.rx_signal.im),
  };
  const grid = demodulate(signal, config);
  assert.ok(largestDifference(grid, vector.rx_grid) < MAX_DEVIATION);
});

test("a changed angle changes the signal, so the comparisons above can fail", () => {
  const alpha = Float64Array.from(vector.alpha);
  alpha[0] += 0.1;
  const angles = { alpha, phi: Float64Array.from(vector.phi) };
  const taps = generateTaps(Float64Array.from(vector.pdp), vector.doppler_hz, angles, config);
  const output = applyChannel(vector.tx_signal, taps);
  assert.ok(largestDifference(output, vector.rx_signal) > 1e-4);
});

test("the channel has a mean power of 1 per resource element, within 2 %", () => {
  const random = new Random(2);
  const pdp = exponentialPdp(1.5, oneSymbol);
  let power = 0;
  const nSlots = 3000;
  for (let slot = 0; slot < nSlots; slot++) {
    const angles = drawAngles(pdp.length, oneSymbol, random);
    const grid = frequencyResponse(generateTaps(pdp, 300, angles, oneSymbol), oneSymbol);
    for (let i = 0; i < grid.re.length; i++) {
      power += grid.re[i] ** 2 + grid.im[i] ** 2;
    }
  }
  const mean = power / (nSlots * config.n_subcarriers);
  assert.ok(Math.abs(mean - 1) < 0.02, `mean power ${mean}`);
});

test("every tap has the power of the delay profile, within 5 %", () => {
  const random = new Random(3);
  const pdp = exponentialPdp(0.6, oneSymbol);
  const power = new Float64Array(pdp.length);
  const nSlots = 3000;
  for (let slot = 0; slot < nSlots; slot++) {
    const taps = generateTaps(pdp, 0, drawAngles(pdp.length, oneSymbol, random), oneSymbol);
    for (let l = 0; l < pdp.length; l++) {
      const first = l * taps.nSamples;
      power[l] += taps.re[first] ** 2 + taps.im[first] ** 2;
    }
  }
  pdp.forEach((p, l) => {
    const ratio = power[l] / nSlots / p;
    assert.ok(Math.abs(ratio - 1) < 0.05, `tap ${l}: ratio ${ratio}`);
  });
});

test("the time correlation of a tap follows J0 of the Jakes model, within 0.05", () => {
  const random = new Random(4);
  const dopplerHz = 300;
  const pdp = Float64Array.of(1);
  const lags = [0, 137, 411, 959, 1781]; // 0, 1, 3, 7 and 13 OFDM symbols
  const sums = new Float64Array(lags.length);
  const nSlots = 3000;
  for (let slot = 0; slot < nSlots; slot++) {
    const taps = generateTaps(pdp, dopplerHz, drawAngles(1, config, random), config);
    lags.forEach((lag, i) => {
      sums[i] += taps.re[0] * taps.re[lag] + taps.im[0] * taps.im[lag];
    });
  }
  lags.forEach((lag, i) => {
    const measured = sums[i] / nSlots;
    const expected = besselJ0((2 * Math.PI * dopplerHz * lag) / config.sample_rate_hz);
    assert.ok(Math.abs(measured - expected) < 0.05, `lag ${lag}: ${measured} vs ${expected}`);
  });
});

test("the noise has the requested variance, within 2 %", () => {
  const random = new Random(5);
  const noiseVariance = 0.1;
  const noise = addNoise(makeComplex(40000), noiseVariance, random);
  let power = 0;
  for (let n = 0; n < noise.re.length; n++) {
    power += noise.re[n] ** 2 + noise.im[n] ** 2;
  }
  const ratio = power / noise.re.length / noiseVariance;
  assert.ok(Math.abs(ratio - 1) < 0.02, `ratio ${ratio}`);
});

test("without Doppler and within the cyclic prefix, Y = H X + noise on every subcarrier", () => {
  const link = makeLink(config);
  const random = new Random(6);
  const slot = simulateSlot(link, 20, 0.3, 0, random);
  let errorPower = 0;
  const nElements = slot.rxGrid.re.length;
  for (let i = 0; i < nElements; i++) {
    const predictedRe = slot.hTrue.re[i] * slot.txGrid.re[i] - slot.hTrue.im[i] * slot.txGrid.im[i];
    const predictedIm = slot.hTrue.re[i] * slot.txGrid.im[i] + slot.hTrue.im[i] * slot.txGrid.re[i];
    errorPower += (slot.rxGrid.re[i] - predictedRe) ** 2 + (slot.rxGrid.im[i] - predictedIm) ** 2;
  }
  // Noise variance per resource element = 1 / SNR, because the FFT keeps the variance.
  const ratio = errorPower / nElements / slot.noiseVariance;
  assert.ok(Math.abs(ratio - 1) < 0.15, `ratio ${ratio}`);
});

test("the same seed gives the same slot, another seed another one", () => {
  const link = makeLink(config);
  const first = simulateSlot(link, 15, 0.5, 100, new Random(7));
  const again = simulateSlot(link, 15, 0.5, 100, new Random(7));
  const other = simulateSlot(link, 15, 0.5, 100, new Random(8));
  assert.equal(largestDifference(first.rxGrid, again.rxGrid), 0);
  assert.ok(largestDifference(first.rxGrid, other.rxGrid) > 0.01);
});

test("the random numbers have the right mean and variance", () => {
  const random = new Random(9);
  const n = 100000;
  let uniformSum = 0;
  let normalSum = 0;
  let normalSquares = 0;
  for (let i = 0; i < n; i++) {
    const u = random.uniform();
    assert.ok(u >= 0 && u < 1);
    uniformSum += u;
    const x = random.normal();
    normalSum += x;
    normalSquares += x * x;
  }
  assert.ok(Math.abs(uniformSum / n - 0.5) < 0.005);
  assert.ok(Math.abs(normalSum / n) < 0.02);
  assert.ok(Math.abs(normalSquares / n - 1) < 0.02);
});
