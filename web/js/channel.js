// Time-varying tapped-delay-line Rayleigh channel with an exponential power delay profile, as in
// src/ofdm/channel.py. Each tap gain is a sum of sinusoids with random arrival angles (Clarke's
// model), whose Doppler spectrum approximates the Jakes spectrum.

import { makeComplex } from "./ofdm.js";

// Sample-spaced exponential power delay profile, truncated at the floor, with a sum of 1.
export function exponentialPdp(delaySpreadUs, config) {
  if (delaySpreadUs <= 0) {
    return Float64Array.of(1);
  }
  const { max_taps: maxTaps, pdp_floor_db: floorDb } = config.channel;
  const tapsPerDecay = delaySpreadUs * 1e-6 * config.sample_rate_hz;
  const span = tapsPerDecay * Math.log(10 ** (-floorDb / 10));
  const nTaps = Math.min(maxTaps, Math.floor(span) + 1);
  const pdp = Float64Array.from({ length: nTaps }, (_, l) => Math.exp(-l / tapsPerDecay));
  const sum = pdp.reduce((a, b) => a + b, 0);
  return pdp.map((p) => p / sum);
}

// Arrival angle and start phase of every sinusoid, in the order tap, sinusoid.
export function drawAngles(nTaps, config, random) {
  const nValues = nTaps * config.channel.n_sinusoids;
  const draw = () => Float64Array.from({ length: nValues }, () => 2 * Math.PI * random.uniform());
  return { alpha: draw(), phi: draw() };
}

// h_l[n] = sqrt(p_l / M) * sum_m exp(j (w_m n + phi_m)) with w_m = 2 pi f_D cos(alpha_m) / f_s.
// Returns {re, im} in the order tap, sample. Each sinusoid is rotated from sample to sample, which
// needs far fewer sines and cosines than evaluating every sample on its own.
export function generateTaps(pdp, dopplerHz, angles, config) {
  const nSamples = config.n_symbols * (config.n_fft + config.cp_length);
  const nSinusoids = config.channel.n_sinusoids;
  const nTaps = pdp.length;
  const taps = makeComplex(nTaps * nSamples);
  for (let l = 0; l < nTaps; l++) {
    const base = l * nSamples;
    for (let m = 0; m < nSinusoids; m++) {
      const index = l * nSinusoids + m;
      const w = (2 * Math.PI * dopplerHz * Math.cos(angles.alpha[index])) / config.sample_rate_hz;
      const stepRe = Math.cos(w);
      const stepIm = Math.sin(w);
      let re = Math.cos(angles.phi[index]);
      let im = Math.sin(angles.phi[index]);
      for (let n = 0; n < nSamples; n++) {
        taps.re[base + n] += re;
        taps.im[base + n] += im;
        const nextRe = re * stepRe - im * stepIm;
        im = re * stepIm + im * stepRe;
        re = nextRe;
      }
    }
    const gain = Math.sqrt(pdp[l] / nSinusoids);
    for (let n = 0; n < nSamples; n++) {
      taps.re[base + n] *= gain;
      taps.im[base + n] *= gain;
    }
  }
  return { ...taps, nTaps, nSamples };
}

// y[n] = sum_l h_l[n] x[n - l]. Samples before the slot are zero.
export function applyChannel(signal, taps) {
  const { nTaps, nSamples } = taps;
  const output = makeComplex(nSamples);
  for (let l = 0; l < nTaps; l++) {
    const base = l * nSamples;
    for (let n = l; n < nSamples; n++) {
      const hr = taps.re[base + n];
      const hi = taps.im[base + n];
      const xr = signal.re[n - l];
      const xi = signal.im[n - l];
      output.re[n] += hr * xr - hi * xi;
      output.im[n] += hr * xi + hi * xr;
    }
  }
  return output;
}

// The true channel per resource element. Each tap is averaged over the FFT window of every symbol.
// For a channel that varies within a symbol this is the diagonal of the inter-carrier interference
// matrix; the rest of that matrix acts as additional noise.
export function frequencyResponse(taps, config) {
  const { n_fft: nFft, cp_length: cp, n_symbols: nSymbols, n_subcarriers: nSubcarriers } = config;
  const bins = config.subcarrier_bins;
  const symbolLength = nFft + cp;
  const { nTaps, nSamples } = taps;
  const grid = makeComplex(nSymbols * nSubcarriers);
  for (let l = 0; l < nTaps; l++) {
    const base = l * nSamples;
    for (let s = 0; s < nSymbols; s++) {
      let meanRe = 0;
      let meanIm = 0;
      const first = base + s * symbolLength + cp;
      for (let n = 0; n < nFft; n++) {
        meanRe += taps.re[first + n];
        meanIm += taps.im[first + n];
      }
      meanRe /= nFft;
      meanIm /= nFft;
      for (let k = 0; k < nSubcarriers; k++) {
        const angle = (-2 * Math.PI * l * bins[k]) / nFft;
        const steerRe = Math.cos(angle);
        const steerIm = Math.sin(angle);
        grid.re[s * nSubcarriers + k] += meanRe * steerRe - meanIm * steerIm;
        grid.im[s * nSubcarriers + k] += meanRe * steerIm + meanIm * steerRe;
      }
    }
  }
  return grid;
}

// Complex white Gaussian noise with the given variance.
export function addNoise(signal, noiseVariance, random) {
  const std = Math.sqrt(noiseVariance / 2);
  const output = makeComplex(signal.re.length);
  for (let n = 0; n < signal.re.length; n++) {
    output.re[n] = signal.re[n] + std * random.normal();
    output.im[n] = signal.im[n] + std * random.normal();
  }
  return output;
}
