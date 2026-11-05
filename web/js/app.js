// The page: reads the sliders, asks the worker for results and draws them.

import {
  cellAt,
  countOutside,
  drawConstellation,
  drawHeatmap,
  drawScale,
  readRamp,
} from "./plots.js";
import { LEVEL_DB, regionOf, verdict } from "./statistics.js";

const PRESETS = {
  training: { snr: 15, delay: 0.3, doppler: 100 },
  unseen: { snr: 15, delay: 1.1, doppler: 450 },
  fails: { snr: 20, delay: 0.1, doppler: 600 },
};
const ESTIMATOR_NAMES = { ls: "LS + linear", cnn: "CNN (int8)", perfect: "Perfect channel knowledge" };

const element = (id) => document.getElementById(id);
const sliders = { snr: element("snr"), delay: element("delay"), doppler: element("doppler") };

let config = null;
let worker = null;
let requestId = 0;
let seed = 1;
let view = null;

function formatDb(value) {
  return `${value.toFixed(1).replace("-", "−")} dB`;
}

function formatBer(value) {
  if (value === 0) {
    return "0";
  }
  return value >= 0.01 ? value.toFixed(3) : value.toExponential(1);
}

function settings() {
  return {
    snrDb: Number(sliders.snr.value),
    delaySpreadUs: Number(sliders.delay.value),
    dopplerHz: Number(sliders.doppler.value),
  };
}

function showSliderValues() {
  const { snrDb, delaySpreadUs, dopplerHz } = settings();
  element("snr-value").textContent = `${snrDb} dB`;
  element("delay-value").textContent = `${delaySpreadUs.toFixed(1)} µs`;
  element("doppler-value").textContent = `${dopplerHz} Hz`;
}

function showRegion() {
  const { delaySpreadUs, dopplerHz } = settings();
  const notice = element("region-notice");
  const region = regionOf(config, delaySpreadUs, dopplerHz);
  const a = config.regions.A;
  const range = `${a.delay_spread_us[0]}–${a.delay_spread_us[1]} µs and ${a.doppler_hz[0]}–${a.doppler_hz[1]} Hz`;
  notice.dataset.status = region === "A" ? "good" : "warning";
  element("region-icon").textContent = region === "A" ? "✓" : "!";
  element("region-text").textContent =
    region === "A"
      ? `Inside the training range (${range}).`
      : `Outside the training range (${range}). The CNN never saw such channels and can be worse than its own input.`;
}

function request() {
  requestId += 1;
  element("result").classList.add("is-stale");
  worker.postMessage({ id: requestId, seed, ...settings() });
}

function update() {
  showSliderValues();
  showRegion();
  request();
}

function showSummary(summary, finished) {
  const { gainDb, slots } = summary;
  const state = verdict(gainDb);
  element("gain-value").textContent = `${gainDb >= 0 ? "+" : "−"}${Math.abs(gainDb).toFixed(1)} dB`;
  const tile = element("gain-tile");
  tile.dataset.status = { better: "good", worse: "critical", level: "neutral" }[state];
  element("gain-icon").textContent = { better: "✓", worse: "✕", level: "=" }[state];
  element("gain-text").textContent = {
    better: "The CNN is closer to the true channel than LS + linear.",
    worse: "The CNN is further from the true channel than LS + linear.",
    level: `Within ${LEVEL_DB} dB: the two are about level.`,
  }[state];
  element("slot-count").textContent = finished
    ? `Mean over ${slots} simulated slots.`
    : `Mean over ${slots} simulated slots so far.`;

  for (const name of ["ls", "cnn", "perfect"]) {
    element(`nmse-${name}`).textContent = name === "perfect" ? "–" : formatDb(summary[name].nmseDb);
    element(`ber-${name}`).textContent = formatBer(summary[name].ber);
  }
}

function drawView() {
  if (view === null) {
    return;
  }
  for (const name of ["ls", "cnn", "perfect"]) {
    const data = view.constellations[name];
    drawConstellation(element(`constellation-${name}`), data);
    const wrong = data.wrong.reduce((sum, value) => sum + value, 0);
    const outside = countOutside(data);
    const beyond = outside > 0 ? `, ${outside} outside the plot` : "";
    element(`constellation-${name}-note`).textContent = `${wrong} bit errors in this slot${beyond}`;
    element(`constellation-${name}`).setAttribute(
      "aria-label",
      `Received symbols after equalization with ${ESTIMATOR_NAMES[name]}: ${wrong} bit errors in this slot.`,
    );
  }

  const blue = readRamp("--ramp-magnitude");
  const orange = readRamp("--ramp-error");
  const { truth, ls, cnn } = view.magnitudes;
  const magnitudeMax = Math.max(...truth, ...ls, ...cnn);
  const errorMax = Math.max(...view.errors.ls, ...view.errors.cnn);
  drawHeatmap(element("map-truth"), truth, magnitudeMax, blue);
  drawHeatmap(element("map-ls"), ls, magnitudeMax, blue);
  drawHeatmap(element("map-cnn"), cnn, magnitudeMax, blue);
  drawHeatmap(element("map-error-ls"), view.errors.ls, errorMax, orange);
  drawHeatmap(element("map-error-cnn"), view.errors.cnn, errorMax, orange);
  drawScale(element("scale-magnitude"), blue);
  drawScale(element("scale-error"), orange);
  element("scale-magnitude-max").textContent = magnitudeMax.toFixed(1);
  element("scale-error-max").textContent = errorMax.toFixed(1);
}

function onMessage({ data }) {
  if (data.type === "ready") {
    config = data.config;
    update();
    return;
  }
  if (data.type === "error") {
    showFailure(data.message);
    return;
  }
  if (data.id !== requestId) {
    return; // a result for settings that have changed in the meantime
  }
  if (data.type === "slot") {
    view = data.view;
    element("result").classList.remove("is-stale");
    drawView();
  }
  showSummary(data.summary, data.finished);
  if (data.msPerSlot !== null && data.msPerSlot !== undefined) {
    element("time-value").textContent = `${data.msPerSlot.toFixed(1)} ms`;
  }
}

function showFailure(message) {
  element("failure").hidden = false;
  element("failure-text").textContent = message;
}

function setUpTooltips() {
  const tooltip = element("tooltip");
  const maps = {
    "map-truth": () => view.magnitudes.truth,
    "map-ls": () => view.magnitudes.ls,
    "map-cnn": () => view.magnitudes.cnn,
    "map-error-ls": () => view.errors.ls,
    "map-error-cnn": () => view.errors.cnn,
  };
  for (const [id, values] of Object.entries(maps)) {
    const canvas = element(id);
    canvas.addEventListener("pointermove", (event) => {
      const cell = view === null ? null : cellAt(canvas, event.clientX, event.clientY);
      if (cell === null) {
        tooltip.hidden = true;
        return;
      }
      tooltip.textContent = `Symbol ${cell.symbol}, subcarrier ${cell.subcarrier}: ${values()[cell.index].toFixed(2)}`;
      tooltip.hidden = false;
      const x = Math.min(event.clientX + 12, window.innerWidth - tooltip.offsetWidth - 8);
      tooltip.style.transform = `translate(${x}px, ${event.clientY + 14}px)`;
    });
    canvas.addEventListener("pointerleave", () => {
      tooltip.hidden = true;
    });
  }
}

function setUpControls() {
  for (const slider of Object.values(sliders)) {
    slider.addEventListener("input", update);
  }
  for (const [name, preset] of Object.entries(PRESETS)) {
    element(`preset-${name}`).addEventListener("click", () => {
      sliders.snr.value = preset.snr;
      sliders.delay.value = preset.delay;
      sliders.doppler.value = preset.doppler;
      update();
    });
  }
  element("new-slot").addEventListener("click", () => {
    seed += 1;
    request();
  });
  new ResizeObserver(drawView).observe(element("result"));
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawView);
}

function start() {
  showSliderValues();
  setUpControls();
  setUpTooltips();
  try {
    worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  } catch (error) {
    showFailure(String(error));
    return;
  }
  worker.onmessage = onMessage;
  worker.onerror = (event) => showFailure(event.message);
}

start();
