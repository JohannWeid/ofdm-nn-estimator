// The whole chain over many slots, the same way scripts/compare_quantization.py evaluates it in Python:
// the delay spread and Doppler of each slot are drawn from a region, NMSE sums over all slots.

import { readFileSync } from "node:fs";

import { makeReceiver, receiveSlot } from "../js/estimators.js";
import { makeLink, simulateSlot } from "../js/link.js";
import { Random } from "../js/random.js";

const REPO = new URL("../../", import.meta.url);

export function loadChain() {
  const web = new URL("web/model/", REPO);
  const config = JSON.parse(readFileSync(new URL("link.json", web), "utf8"));
  const model = JSON.parse(readFileSync(new URL("cnn_A_log_nmse_int8.json", web), "utf8"));
  const link = makeLink(config);
  return { config, link, receiver: makeReceiver(link, model) };
}

// NMSE in dB and BER per estimator at one SNR.
export function evaluate({ config, link, receiver }, regionName, snrDb, nSlots, seed) {
  const region = config.regions[regionName];
  const random = new Random(seed);
  const draw = ([low, high]) => low + (high - low) * random.uniform();
  const totals = { ls: { error: 0, bits: 0 }, cnn: { error: 0, bits: 0 } };
  let channelEnergy = 0;
  let nBits = 0;
  for (let n = 0; n < nSlots; n++) {
    const delay = draw(region.delay_spread_us);
    const doppler = draw(region.doppler_hz);
    const result = receiveSlot(receiver, simulateSlot(link, snrDb, delay, doppler, random), snrDb);
    for (const name of ["ls", "cnn"]) {
      totals[name].error += result[name].squaredError;
      totals[name].bits += result[name].bitErrors;
    }
    channelEnergy += result.channelEnergy;
    nBits += result.nBits;
  }
  const summary = (total) => ({
    nmseDb: 10 * Math.log10(total.error / channelEnergy),
    ber: total.bits / nBits,
  });
  return { ls: summary(totals.ls), cnn: summary(totals.cnn) };
}

// The results of the Python chain on the test slots of the README, by region, estimator and SNR.
export function readPythonResults() {
  const text = readFileSync(new URL("results/quantization.csv", REPO), "utf8");
  const results = new Map();
  for (const line of text.trim().split("\n").slice(1)) {
    const [region, snr, estimator, , nmseDb, ber] = line.trim().split(",");
    const name = { "LS + linear": "ls", "standard int8": "cnn" }[estimator];
    if (name) {
      results.set(`${region}/${name}/${Number(snr)}`, { nmseDb: Number(nmseDb), ber: Number(ber) });
    }
  }
  return results;
}
