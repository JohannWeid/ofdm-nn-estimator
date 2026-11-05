// Drawing on canvas, without a library. Colors come from the CSS custom properties of the page, so
// light and dark mode are defined in one place.

const N_SYMBOLS = 14;
const N_SUBCARRIERS = 72;
const CONSTELLATION_RANGE = 2; // the plot shows -2 to 2 on both axes
const IDEAL = Math.SQRT1_2;

export function cssColor(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Sizes the canvas for the screen and returns a context drawing in CSS pixels.
function prepare(canvas) {
  const { width, height } = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, width, height };
}

export function drawConstellation(canvas, { re, im, wrong }) {
  const { context, width, height } = prepare(canvas);
  const toX = (value) => ((value + CONSTELLATION_RANGE) / (2 * CONSTELLATION_RANGE)) * width;
  const toY = (value) => ((CONSTELLATION_RANGE - value) / (2 * CONSTELLATION_RANGE)) * height;

  context.strokeStyle = cssColor("--grid");
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(toX(0), 0);
  context.lineTo(toX(0), height);
  context.moveTo(0, toY(0));
  context.lineTo(width, toY(0));
  context.stroke();

  context.globalAlpha = 0.6;
  context.fillStyle = cssColor("--mark");
  for (let i = 0; i < re.length; i++) {
    if (wrong[i] === 0) {
      context.beginPath();
      context.arc(toX(re[i]), toY(im[i]), 1.6, 0, 2 * Math.PI);
      context.fill();
    }
  }
  context.globalAlpha = 1;

  // Wrong decisions differ by shape as well as by color.
  context.strokeStyle = cssColor("--critical");
  context.lineWidth = 1.5;
  context.beginPath();
  for (let i = 0; i < re.length; i++) {
    if (wrong[i] > 0) {
      const x = toX(re[i]);
      const y = toY(im[i]);
      context.moveTo(x - 3, y - 3);
      context.lineTo(x + 3, y + 3);
      context.moveTo(x - 3, y + 3);
      context.lineTo(x + 3, y - 3);
    }
  }
  context.stroke();

  context.strokeStyle = cssColor("--ink");
  context.lineWidth = 1.5;
  for (const x of [-IDEAL, IDEAL]) {
    for (const y of [-IDEAL, IDEAL]) {
      context.beginPath();
      context.arc(toX(x), toY(y), 5, 0, 2 * Math.PI);
      context.stroke();
    }
  }
}

// How many points lie outside the plot? A deep fade can throw a point far away.
export function countOutside({ re, im }) {
  let outside = 0;
  for (let i = 0; i < re.length; i++) {
    if (Math.abs(re[i]) > CONSTELLATION_RANGE || Math.abs(im[i]) > CONSTELLATION_RANGE) {
      outside += 1;
    }
  }
  return outside;
}

function hexToRgb(hex) {
  const value = Number.parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

// A color from a list of hex steps, for a value between 0 and 1.
function rampColor(steps, share) {
  const position = Math.min(Math.max(share, 0), 1) * (steps.length - 1);
  const lower = Math.min(Math.floor(position), steps.length - 2);
  const [r0, g0, b0] = steps[lower];
  const [r1, g1, b1] = steps[lower + 1];
  const t = position - lower;
  return `rgb(${r0 + (r1 - r0) * t}, ${g0 + (g1 - g0) * t}, ${b0 + (b1 - b0) * t})`;
}

export function readRamp(name) {
  return cssColor(name).split(",").map((hex) => hexToRgb(hex.trim()));
}

export function drawHeatmap(canvas, values, maximum, ramp) {
  const { context, width, height } = prepare(canvas);
  const cellWidth = width / N_SUBCARRIERS;
  const cellHeight = height / N_SYMBOLS;
  for (let s = 0; s < N_SYMBOLS; s++) {
    for (let k = 0; k < N_SUBCARRIERS; k++) {
      context.fillStyle = rampColor(ramp, values[s * N_SUBCARRIERS + k] / maximum);
      // Slightly larger than a cell, so that no thin gaps appear when the sizes are not whole pixels.
      context.fillRect(k * cellWidth, s * cellHeight, cellWidth + 0.5, cellHeight + 0.5);
    }
  }
}

export function drawScale(canvas, ramp) {
  const { context, width, height } = prepare(canvas);
  for (let x = 0; x < width; x++) {
    context.fillStyle = rampColor(ramp, x / (width - 1));
    context.fillRect(x, 0, 1.5, height);
  }
}

// Which cell of a heatmap lies under a pointer position?
export function cellAt(canvas, clientX, clientY) {
  const box = canvas.getBoundingClientRect();
  const k = Math.floor(((clientX - box.left) / box.width) * N_SUBCARRIERS);
  const s = Math.floor(((clientY - box.top) / box.height) * N_SYMBOLS);
  if (k < 0 || k >= N_SUBCARRIERS || s < 0 || s >= N_SYMBOLS) {
    return null;
  }
  return { symbol: s, subcarrier: k, index: s * N_SUBCARRIERS + k };
}
