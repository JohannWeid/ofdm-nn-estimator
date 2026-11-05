// The whole JavaScript chain against the Python results, with many slots, so too slow for the CI.
//
// Usage, from the repository root: node web/tests/compare_with_python.mjs [slots per SNR]
// Writes results/web_chain.csv and fails if an NMSE is more than 0.3 dB off.

import { writeFileSync } from "node:fs";

import { evaluate, loadChain, readPythonResults } from "./chain.mjs";

const MAX_DIFFERENCE_DB = 0.3;
const SNR_DB = [0, 5, 10, 15, 20, 25, 30];
const SEED = { A: 201, B: 301 };
const NAMES = { ls: "LS + linear", cnn: "CNN int8" };

const nSlots = Number(process.argv[2] ?? 1000);
const chain = loadChain();
const python = readPythonResults();
const rows = ["region,snr_db,estimator,nmse_db,python_nmse_db,difference_db,ber,python_ber"];
let worst = 0;
for (const region of ["A", "B"]) {
  console.log(`\nRegion ${region}, ${nSlots} slots per SNR. JavaScript, Python, difference in dB; BER`);
  for (const snr of SNR_DB) {
    const result = evaluate(chain, region, snr, nSlots, SEED[region] + snr);
    for (const name of ["ls", "cnn"]) {
      const js = result[name];
      const py = python.get(`${region}/${name}/${snr}`);
      const difference = js.nmseDb - py.nmseDb;
      worst = Math.max(worst, Math.abs(difference));
      rows.push(
        [region, snr, NAMES[name], js.nmseDb, py.nmseDb, difference, js.ber, py.ber].join(","),
      );
      console.log(
        `${String(snr).padStart(3)} dB  ${NAMES[name].padEnd(12)}` +
          `${js.nmseDb.toFixed(2).padStart(8)}${py.nmseDb.toFixed(2).padStart(8)}` +
          `${difference.toFixed(2).padStart(7)}` +
          `${js.ber.toExponential(2).padStart(11)}${py.ber.toExponential(2).padStart(11)}`,
      );
    }
  }
}
writeFileSync(new URL("../../results/web_chain.csv", import.meta.url), `${rows.join("\n")}\n`);
console.log(`\nlargest NMSE difference: ${worst.toFixed(3)} dB, limit ${MAX_DIFFERENCE_DB} dB`);
if (worst > MAX_DIFFERENCE_DB) {
  process.exitCode = 1;
}
