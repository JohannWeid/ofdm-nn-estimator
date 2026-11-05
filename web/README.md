# Browser demo (Phase 5)

A static page with sliders for SNR, delay spread and Doppler, and constellations and channel pictures
for classical and neural estimation. No install, no backend, no packages: everything runs in plain
JavaScript on the viewer's device.

To look at it locally, serve this folder, because JavaScript modules and workers do not run from
`file://`. From the repository root:

```bash
python -m http.server 8123 --directory web
```

Then open <http://localhost:8123>.

## The code

- [`js/network.js`](js/network.js): the int8 network, compared with the reference from ONNX Runtime in
  `cpp/testdata/`.
- [`js/link.js`](js/link.js), [`js/channel.js`](js/channel.js), [`js/ofdm.js`](js/ofdm.js) and
  [`js/random.js`](js/random.js): the simulator of one slot. With the same bits and angles it gives the
  same signals as the Python simulator. Its own random numbers are checked by their statistics, because
  JavaScript cannot reproduce NumPy's generator.
- [`js/estimators.js`](js/estimators.js): LS + linear, the network input, equalization and bit errors.
  On one noisy slot it gives the same estimates and the same number of bit errors as Python.
- [`js/statistics.js`](js/statistics.js): sums over many slots, the region notice and the verdict.
- [`js/worker.js`](js/worker.js) runs all of that away from the main thread, so that the sliders stay
  smooth. [`js/app.js`](js/app.js) and [`js/plots.js`](js/plots.js) are the page itself.

## Tests

`python scripts/export_web_data.py` writes the parameters to `model/` and the comparison data to
`tests/vectors/`. The tests only need Node. From the repository root:

```bash
node --test "web/tests/*.test.mjs"
```

The whole chain over many slots takes a few minutes and is not part of the CI. It compares with
`results/quantization.csv` and writes `results/web_chain.csv`:

```bash
node web/tests/compare_with_python.mjs
```
