// Runs the simulation away from the main thread, so that the sliders stay smooth. A new request
// replaces the running one between two slots: only the latest settings are ever computed.

import { makeReceiver, receiveSlot } from "./estimators.js";
import { makeLink, simulateSlot } from "./link.js";
import { Random } from "./random.js";
import { addSlot, makeTotals, summarize, wrongBits } from "./statistics.js";

const MAX_SLOTS = 200;
const REPORT_EVERY_MS = 120;

let link = null;
let receiver = null;
let job = null;
let stepPending = false;

function loadJson(path) {
  return fetch(new URL(path, import.meta.url)).then((response) => {
    if (!response.ok) {
      throw new Error(`could not load ${path}: ${response.status}`);
    }
    return response.json();
  });
}

function magnitude(grid) {
  return Float32Array.from(grid.re, (re, i) => Math.hypot(re, grid.im[i]));
}

function errorMagnitude(estimate, truth) {
  return Float32Array.from(estimate.re, (re, i) =>
    Math.hypot(re - truth.re[i], estimate.im[i] - truth.im[i]),
  );
}

// Everything needed to draw one slot, as typed arrays that can be handed over without copying.
function makeView(slot, result) {
  const constellation = (name) => ({
    re: Float32Array.from(result[name].equalized.re),
    im: Float32Array.from(result[name].equalized.im),
    wrong: wrongBits(result[name].equalized, slot.bits),
  });
  return {
    constellations: { ls: constellation("ls"), cnn: constellation("cnn"), perfect: constellation("perfect") },
    magnitudes: {
      truth: magnitude(slot.hTrue),
      ls: magnitude(result.ls.estimate),
      cnn: magnitude(result.cnn.estimate),
    },
    errors: {
      ls: errorMagnitude(result.ls.estimate, slot.hTrue),
      cnn: errorMagnitude(result.cnn.estimate, slot.hTrue),
    },
  };
}

function transferables(view) {
  const { constellations, magnitudes, errors } = view;
  const buffers = [...Object.values(magnitudes), ...Object.values(errors)].map((a) => a.buffer);
  for (const { re, im, wrong } of Object.values(constellations)) {
    buffers.push(re.buffer, im.buffer, wrong.buffer);
  }
  return buffers;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function report(current, type, extra = {}) {
  const times = current.times;
  self.postMessage({
    type,
    id: current.id,
    summary: summarize(current.totals),
    // The first runs are slow until the JavaScript engine has optimized the code.
    msPerSlot: times.length > 10 ? median(times.slice(5)) : null,
    finished: current.totals.slots >= MAX_SLOTS,
    ...extra,
  });
}

function step() {
  stepPending = false;
  const current = job;
  if (current === null || current.totals.slots >= MAX_SLOTS) {
    return;
  }
  const { snrDb, delaySpreadUs, dopplerHz, random } = current;
  const start = performance.now();
  const slot = simulateSlot(link, snrDb, delaySpreadUs, dopplerHz, random);
  const result = receiveSlot(receiver, slot, snrDb);
  current.times.push(performance.now() - start);
  addSlot(current.totals, result);

  const first = current.totals.slots === 1;
  const now = performance.now();
  const last = current.totals.slots >= MAX_SLOTS;
  if (first) {
    const view = makeView(slot, result);
    self.postMessage(
      { type: "slot", id: current.id, view, summary: summarize(current.totals), finished: false },
      transferables(view),
    );
    current.lastReport = now;
  } else if (last || now - current.lastReport > REPORT_EVERY_MS) {
    report(current, "update");
    current.lastReport = now;
  }
  schedule();
}

function schedule() {
  if (!stepPending && job !== null && job.totals.slots < MAX_SLOTS) {
    stepPending = true;
    setTimeout(step, 0); // lets a newer request arrive between two slots
  }
}

const ready = Promise.all([
  loadJson("../model/link.json"),
  loadJson("../model/cnn_A_log_nmse_int8.json"),
]).then(([config, model]) => {
  link = makeLink(config);
  receiver = makeReceiver(link, model);
  self.postMessage({ type: "ready", config });
});

ready.catch((error) => self.postMessage({ type: "error", message: String(error) }));

self.onmessage = ({ data }) => {
  ready.then(() => {
    job = {
      ...data,
      random: new Random(data.seed),
      totals: makeTotals(),
      times: [],
      lastReport: 0,
    };
    schedule();
  });
};
