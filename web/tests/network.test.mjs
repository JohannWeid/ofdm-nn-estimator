// The JavaScript network against the reference from ONNX Runtime, see scripts/make_test_vectors.py.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { INPUT_SIZE, OUTPUT_SIZE, prepare, roundHalfEven, runCorrection } from "../js/network.js";

const REPO = new URL("../../", import.meta.url);
const MODELS = ["cnn_A_log_nmse_int8", "cnn_A_small_int8"];
const N_SLOTS = 5;

// Raw float32 in little endian, which is what x86 and ARM use in memory.
function readVector(name) {
  const bytes = readFileSync(new URL(`cpp/testdata/${name}.bin`, REPO));
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
}

function readModel(name) {
  return JSON.parse(readFileSync(new URL(`web/model/${name}.json`, REPO), "utf8"));
}

function runAllSlots(model) {
  const input = readVector("input");
  const network = prepare(model);
  const output = new Float32Array(N_SLOTS * OUTPUT_SIZE);
  for (let slot = 0; slot < N_SLOTS; slot++) {
    runCorrection(
      network,
      input.subarray(slot * INPUT_SIZE, (slot + 1) * INPUT_SIZE),
      output.subarray(slot * OUTPUT_SIZE, (slot + 1) * OUTPUT_SIZE),
    );
  }
  return { output, network };
}

function countDifferent(a, b) {
  let count = 0;
  for (let i = 0; i < a.length; i++) {
    count += a[i] === b[i] ? 0 : 1;
  }
  return count;
}

for (const name of MODELS) {
  test(`${name} is bit-identical to ONNX Runtime`, () => {
    const expected = readVector(name);
    assert.equal(expected.length, N_SLOTS * OUTPUT_SIZE);
    assert.equal(countDifferent(runAllSlots(readModel(name)).output, expected), 0);
  });

  test(`${name} has to clip the input in the last test slot`, () => {
    const { network } = runAllSlots(readModel(name));
    const extremes = network.quantizedInput.filter((q) => q === 127 || q === -128);
    assert.ok(extremes.length > 0);
  });
}

test("a changed weight changes the result, so the comparison above can fail", () => {
  const model = readModel(MODELS[0]);
  model.layers[1].weights[0] += 1;
  const different = countDifferent(runAllSlots(model).output, readVector(MODELS[0]));
  assert.ok(different > 0);
});

test("a slot gives the same result after another slot has used the working memory", () => {
  const input = readVector("input");
  const network = prepare(readModel(MODELS[0]));
  const first = new Float32Array(OUTPUT_SIZE);
  const again = new Float32Array(OUTPUT_SIZE);
  runCorrection(network, input.subarray(0, INPUT_SIZE), first);
  runCorrection(network, input.subarray(4 * INPUT_SIZE), new Float32Array(OUTPUT_SIZE));
  runCorrection(network, input.subarray(0, INPUT_SIZE), again);
  assert.equal(countDifferent(first, again), 0);
});

test("the input stays untouched", () => {
  const input = readVector("input").slice(0, INPUT_SIZE);
  const copy = input.slice();
  runCorrection(prepare(readModel(MODELS[0])), input, new Float32Array(OUTPUT_SIZE));
  assert.equal(countDifferent(input, copy), 0);
});

test("roundHalfEven rounds a half to the even neighbour", () => {
  const cases = [
    [0.5, 0],
    [1.5, 2],
    [2.5, 2],
    [3.5, 4],
    [-0.5, 0],
    [-1.5, -2],
    [-2.5, -2],
    [-3.5, -4],
    [2.4, 2],
    [2.6, 3],
    [-2.4, -2],
    [-2.6, -3],
    [7, 7],
  ];
  for (const [value, expected] of cases) {
    assert.ok(roundHalfEven(value) === expected, `roundHalfEven(${value})`);
  }
});

test("a model with other channel counts is rejected", () => {
  const model = readModel(MODELS[0]);
  model.layers[0].in_channels = 2;
  assert.throws(() => prepare(model), /3 input channels/);
});
