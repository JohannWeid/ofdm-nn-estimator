// What the page says about the results: sums over slots, the region notice and the verdict.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { addSlot, makeTotals, regionOf, summarize, verdict, wrongBits } from "../js/statistics.js";

const config = JSON.parse(readFileSync(new URL("../model/link.json", import.meta.url), "utf8"));

function fakeSlot(lsError, cnnError, energy, bitErrors) {
  return {
    channelEnergy: energy,
    nBits: 1920,
    ls: { squaredError: lsError, bitErrors },
    cnn: { squaredError: cnnError, bitErrors: bitErrors / 2 },
    perfect: { squaredError: 0, bitErrors: 0 },
  };
}

test("the NMSE sums over all slots instead of averaging their ratios", () => {
  const totals = makeTotals();
  addSlot(totals, fakeSlot(1, 0.5, 10, 40));
  addSlot(totals, fakeSlot(9, 4.5, 90, 80));
  const summary = summarize(totals);
  assert.equal(summary.slots, 2);
  assert.ok(Math.abs(summary.ls.nmseDb - -10) < 1e-12); // 10 / 100
  assert.ok(Math.abs(summary.cnn.nmseDb - 10 * Math.log10(5 / 100)) < 1e-12);
  assert.ok(Math.abs(summary.ls.ber - 120 / 3840) < 1e-12);
  assert.ok(Math.abs(summary.gainDb - 10 * Math.log10(2)) < 1e-12);
});

test("the settings fall into the right region, including its edges", () => {
  assert.equal(regionOf(config, 0.1, 0), "A");
  assert.equal(regionOf(config, 0.6, 150), "A");
  assert.equal(regionOf(config, 0.3, 200), null);
  assert.equal(regionOf(config, 0.7, 100), null);
  assert.equal(regionOf(config, 1.1, 450), "B");
  assert.equal(regionOf(config, 0.1, 600), null);
});

test("the verdict only speaks of better or worse beyond the limit", () => {
  assert.equal(verdict(1), "better");
  assert.equal(verdict(-1), "worse");
  assert.equal(verdict(0.1), "level");
  assert.equal(verdict(-0.1), "level");
});

test("wrong bits are counted per data element", () => {
  const equalized = { re: [0.7, -0.7, 0.7], im: [0.7, -0.7, -0.7] };
  const bits = [1, 1, 1, 1, 1, 1];
  assert.deepEqual([...wrongBits(equalized, bits)], [0, 2, 1]);
});
