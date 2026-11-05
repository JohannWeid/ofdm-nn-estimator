// The JavaScript receiver against the Python one: on one noisy slot it has to give the same estimates
// and decisions, and over a few slots the whole chain has to come close to the Python results.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  cnnEstimate,
  countBitErrors,
  energy,
  equalize,
  lsLinear,
  makeReceiver,
  squaredError,
  toInput,
} from "../js/estimators.js";
import { buildGrid, makeLink } from "../js/link.js";
import { makeComplex } from "../js/ofdm.js";
import { evaluate, loadChain, readPythonResults } from "./chain.mjs";

const WEB = new URL("../", import.meta.url);
const config = JSON.parse(readFileSync(new URL("model/link.json", WEB), "utf8"));
const model = JSON.parse(readFileSync(new URL("model/cnn_A_log_nmse_int8.json", WEB), "utf8"));
const vector = JSON.parse(readFileSync(new URL("tests/vectors/simulator_slot.json", WEB), "utf8"));
const MAX_DEVIATION = 1e-6;

const link = makeLink(config);
const receiver = makeReceiver(link, model);
const toComplex = (v) => ({ re: Float64Array.from(v.re), im: Float64Array.from(v.im) });
const rxGrid = toComplex(vector.rx_grid_noisy);
const hTrue = toComplex(vector.h_true);

function largestDifference(a, b) {
  let largest = 0;
  for (let i = 0; i < a.re.length; i++) {
    largest = Math.max(largest, Math.abs(a.re[i] - b.re[i]), Math.abs(a.im[i] - b.im[i]));
  }
  return largest;
}

test("LS + linear gives the same estimate as Python", () => {
  const estimate = lsLinear(receiver, rxGrid);
  assert.ok(largestDifference(estimate, vector.h_ls_linear) < MAX_DEVIATION);
});

test("LS + linear reproduces a channel that is linear in time and frequency", () => {
  const { n_symbols: nSymbols, n_subcarriers: nSubcarriers } = config;
  const truth = makeComplex(nSymbols * nSubcarriers);
  const received = makeComplex(nSymbols * nSubcarriers);
  const transmitted = buildGrid(link, new Uint8Array(link.nBits));
  for (let s = 0; s < nSymbols; s++) {
    for (let k = 0; k < nSubcarriers; k++) {
      const i = s * nSubcarriers + k;
      truth.re[i] = 0.5 + 0.03 * s - 0.01 * k;
      truth.im[i] = -0.2 + 0.02 * s + 0.005 * k;
      received.re[i] = truth.re[i] * transmitted.re[i] - truth.im[i] * transmitted.im[i];
      received.im[i] = truth.re[i] * transmitted.im[i] + truth.im[i] * transmitted.re[i];
    }
  }
  assert.ok(largestDifference(lsLinear(receiver, received), truth) < 1e-12);
});

test("the network input has the estimate in float32 and the SNR as a plane", () => {
  const estimate = toComplex(vector.h_ls_linear);
  const input = toInput(estimate, vector.snr_db, config.snr_scale_db, new Float32Array(3 * 1008));
  assert.equal(input[5], Math.fround(estimate.re[5]));
  assert.equal(input[1008 + 5], Math.fround(estimate.im[5]));
  assert.equal(input[2 * 1008], Math.fround(vector.snr_db / 30));
  assert.equal(input[3 * 1008 - 1], Math.fround(vector.snr_db / 30));
});

test("the CNN gives the same estimate as Python with ONNX Runtime", () => {
  const estimate = cnnEstimate(receiver, lsLinear(receiver, rxGrid), vector.snr_db);
  assert.ok(largestDifference(estimate, vector.h_cnn) < MAX_DEVIATION);
});

test("the CNN is closer to the true channel than its input on this slot", () => {
  const ls = lsLinear(receiver, rxGrid);
  const cnn = cnnEstimate(receiver, ls, vector.snr_db);
  assert.ok(squaredError(cnn, hTrue) < squaredError(ls, hTrue));
});

test("the NMSE matches Python", () => {
  const ls = lsLinear(receiver, rxGrid);
  const cnn = cnnEstimate(receiver, ls, vector.snr_db);
  const db = (estimate) => 10 * Math.log10(squaredError(estimate, hTrue) / energy(hTrue));
  assert.ok(Math.abs(db(ls) - vector.nmse_db.ls_linear) < MAX_DEVIATION);
  assert.ok(Math.abs(db(cnn) - vector.nmse_db.cnn) < MAX_DEVIATION);
});

test("equalization and hard decisions give the same bit errors as Python", () => {
  const ls = lsLinear(receiver, rxGrid);
  const estimates = {
    ls_linear: ls,
    cnn: cnnEstimate(receiver, ls, vector.snr_db),
    perfect: hTrue,
  };
  for (const [name, estimate] of Object.entries(estimates)) {
    const errors = countBitErrors(equalize(link, rxGrid, estimate), vector.bits);
    assert.equal(errors, vector.bit_errors[name], name);
  }
});

test("over a few slots the whole chain comes close to the Python results", () => {
  // The full comparison with 1000 slots per SNR and a limit of 0.3 dB is compare_with_python.mjs.
  const chain = loadChain();
  const python = readPythonResults();
  for (const snr of [0, 15, 30]) {
    const result = evaluate(chain, "A", snr, 60, 400 + snr);
    for (const name of ["ls", "cnn"]) {
      const expected = python.get(`A/${name}/${snr}`).nmseDb;
      const actual = result[name].nmseDb;
      assert.ok(Math.abs(actual - expected) < 1.0, `${name} at ${snr} dB: ${actual} vs ${expected}`);
    }
    assert.ok(result.cnn.nmseDb < result.ls.nmseDb, `CNN behind LS at ${snr} dB in region A`);
  }
});
