// OFDM modulation (IFFT + cyclic prefix) and demodulation, as in src/ofdm/ofdm.py.
//
// Complex data is an object {re, im} of two Float64Arrays. A grid holds n_symbols x n_subcarriers
// values in the order symbol, subcarrier. The FFTs use the same scaling as norm="ortho" in NumPy, so
// white noise keeps its variance per subcarrier.

export function makeComplex(length) {
  return { re: new Float64Array(length), im: new Float64Array(length) };
}

const fftTables = new Map();

function fftTable(n) {
  if (!fftTables.has(n)) {
    if (n < 2 || (n & (n - 1)) !== 0) {
      throw new Error(`the FFT size must be a power of two, not ${n}`);
    }
    const reversed = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      reversed[i] = (reversed[i >> 1] >> 1) | ((i & 1) << (bits - 1));
    }
    const cos = new Float64Array(n / 2);
    const sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      cos[i] = Math.cos((2 * Math.PI * i) / n);
      sin[i] = Math.sin((2 * Math.PI * i) / n);
    }
    fftTables.set(n, { reversed, cos, sin, scale: 1 / Math.sqrt(n) });
  }
  return fftTables.get(n);
}

// Radix-2 FFT in place. inverse = false uses exp(-2j pi k n / N), inverse = true exp(+2j pi k n / N).
export function fft(re, im, inverse = false) {
  const n = re.length;
  const { reversed, cos, sin, scale } = fftTable(n);
  for (let i = 0; i < n; i++) {
    const j = reversed[i];
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  const sign = inverse ? 1 : -1;
  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2;
    const stride = n / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * stride];
        const wi = sign * sin[k * stride];
        const a = start + k;
        const b = a + half;
        const tr = re[b] * wr - im[b] * wi;
        const ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
  for (let i = 0; i < n; i++) {
    re[i] *= scale;
    im[i] *= scale;
  }
}

// Resource grid -> time signal of n_symbols * (n_fft + cp_length) samples.
export function modulate(grid, config) {
  const { n_fft: nFft, cp_length: cp, n_symbols: nSymbols, n_subcarriers: nSubcarriers } = config;
  const bins = config.subcarrier_bins;
  const symbolLength = nFft + cp;
  const signal = makeComplex(nSymbols * symbolLength);
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  for (let s = 0; s < nSymbols; s++) {
    re.fill(0);
    im.fill(0);
    for (let k = 0; k < nSubcarriers; k++) {
      re[bins[k]] = grid.re[s * nSubcarriers + k];
      im[bins[k]] = grid.im[s * nSubcarriers + k];
    }
    fft(re, im, true);
    const start = s * symbolLength;
    for (let n = 0; n < cp; n++) {
      signal.re[start + n] = re[nFft - cp + n];
      signal.im[start + n] = im[nFft - cp + n];
    }
    signal.re.set(re, start + cp);
    signal.im.set(im, start + cp);
  }
  return signal;
}

// Time signal -> resource grid, dropping the cyclic prefix.
export function demodulate(signal, config) {
  const { n_fft: nFft, cp_length: cp, n_symbols: nSymbols, n_subcarriers: nSubcarriers } = config;
  const bins = config.subcarrier_bins;
  const symbolLength = nFft + cp;
  const grid = makeComplex(nSymbols * nSubcarriers);
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  for (let s = 0; s < nSymbols; s++) {
    const start = s * symbolLength + cp;
    re.set(signal.re.subarray(start, start + nFft));
    im.set(signal.im.subarray(start, start + nFft));
    fft(re, im, false);
    for (let k = 0; k < nSubcarriers; k++) {
      grid.re[s * nSubcarriers + k] = re[bins[k]];
      grid.im[s * nSubcarriers + k] = im[bins[k]];
    }
  }
  return grid;
}
