// Sums over many slots and what the page says about them. NMSE sums over all slots, as in Python.

const ESTIMATORS = ["ls", "cnn", "perfect"];

export function makeTotals() {
  const totals = { slots: 0, channelEnergy: 0, nBits: 0 };
  for (const name of ESTIMATORS) {
    totals[name] = { squaredError: 0, bitErrors: 0 };
  }
  return totals;
}

// result comes from receiveSlot in estimators.js.
export function addSlot(totals, result) {
  totals.slots += 1;
  totals.channelEnergy += result.channelEnergy;
  totals.nBits += result.nBits;
  for (const name of ESTIMATORS) {
    totals[name].squaredError += result[name].squaredError;
    totals[name].bitErrors += result[name].bitErrors;
  }
}

export function summarize(totals) {
  const summary = { slots: totals.slots };
  for (const name of ESTIMATORS) {
    summary[name] = {
      nmseDb: 10 * Math.log10(totals[name].squaredError / totals.channelEnergy),
      ber: totals[name].bitErrors / totals.nBits,
    };
  }
  summary.gainDb = summary.ls.nmseDb - summary.cnn.nmseDb;
  return summary;
}

// Which region of the project do these settings belong to? "A" is the training range.
export function regionOf(config, delaySpreadUs, dopplerHz) {
  for (const [name, region] of Object.entries(config.regions)) {
    const [delayLow, delayHigh] = region.delay_spread_us;
    const [dopplerLow, dopplerHigh] = region.doppler_hz;
    if (
      delaySpreadUs >= delayLow &&
      delaySpreadUs <= delayHigh &&
      dopplerHz >= dopplerLow &&
      dopplerHz <= dopplerHigh
    ) {
      return name;
    }
  }
  return null;
}

// How does the gain compare with zero? Below this difference the page calls the two about level.
export const LEVEL_DB = 0.2;

export function verdict(gainDb) {
  if (gainDb > LEVEL_DB) {
    return "better";
  }
  return gainDb < -LEVEL_DB ? "worse" : "level";
}

// The number of wrong bits per data element, for marking wrong decisions in the constellations.
export function wrongBits(equalized, bits) {
  const wrong = new Uint8Array(equalized.re.length);
  for (let i = 0; i < wrong.length; i++) {
    wrong[i] += (equalized.re[i] > 0 ? 1 : 0) === bits[2 * i] ? 0 : 1;
    wrong[i] += (equalized.im[i] > 0 ? 1 : 0) === bits[2 * i + 1] ? 0 : 1;
  }
  return wrong;
}
