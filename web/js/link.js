// The transmitter and the channel of one slot, as in src/ofdm/link.py: bits -> QPSK -> resource grid
// -> OFDM -> channel -> noise -> FFT. The receiver's estimators are in estimators.js.

import {
  addNoise,
  applyChannel,
  drawAngles,
  exponentialPdp,
  frequencyResponse,
  generateTaps,
} from "./channel.js";
import { demodulate, makeComplex, modulate } from "./ofdm.js";

// config is the parsed web/model/link.json from scripts/export_web_data.py.
export function makeLink(config) {
  if (config.modulation !== "qpsk") {
    throw new Error(`only QPSK is implemented, not ${config.modulation}`);
  }
  const { n_symbols: nSymbols, n_subcarriers: nSubcarriers, pilots } = config;
  const nElements = nSymbols * nSubcarriers;
  const isPilot = new Uint8Array(nElements);
  const pilotGrid = makeComplex(nElements);
  pilots.symbol.forEach((symbol, i) => {
    const element = symbol * nSubcarriers + pilots.subcarrier[i];
    isPilot[element] = 1;
    pilotGrid.re[element] = pilots.re[i];
    pilotGrid.im[element] = pilots.im[i];
  });
  const dataElements = new Int32Array(nElements - pilots.symbol.length);
  let next = 0;
  for (let element = 0; element < nElements; element++) {
    if (!isPilot[element]) {
      dataElements[next++] = element;
    }
  }
  return { config, isPilot, pilotGrid, dataElements, nBits: 2 * dataElements.length };
}

// Gray-coded QPSK: bit 0 selects the sign of the real part, bit 1 the sign of the imaginary part.
export function buildGrid(link, bits) {
  const grid = { re: link.pilotGrid.re.slice(), im: link.pilotGrid.im.slice() };
  link.dataElements.forEach((element, i) => {
    grid.re[element] = (2 * bits[2 * i] - 1) * Math.SQRT1_2;
    grid.im[element] = (2 * bits[2 * i + 1] - 1) * Math.SQRT1_2;
  });
  return grid;
}

// One slot through the whole chain. The SNR is defined per resource element: noise variance = 1 / SNR.
export function simulateSlot(link, snrDb, delaySpreadUs, dopplerHz, random) {
  const { config } = link;
  const bits = Uint8Array.from({ length: link.nBits }, () => (random.uniform() < 0.5 ? 0 : 1));
  const txGrid = buildGrid(link, bits);
  const pdp = exponentialPdp(delaySpreadUs, config);
  const taps = generateTaps(pdp, dopplerHz, drawAngles(pdp.length, config, random), config);
  const noiseVariance = 10 ** (-snrDb / 10);
  const received = addNoise(applyChannel(modulate(txGrid, config), taps), noiseVariance, random);
  return {
    bits,
    txGrid,
    rxGrid: demodulate(received, config),
    hTrue: frequencyResponse(taps, config),
    noiseVariance,
    nTaps: pdp.length,
  };
}
