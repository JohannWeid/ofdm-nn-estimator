# Browser demo (Phase 5)

Planned: a static page on GitHub Pages with sliders for SNR, delay spread and Doppler, and
constellations for classical and neural estimation. No install, no backend, no packages.

So far, everything runs in plain JavaScript, without any library:

- [`js/network.js`](js/network.js): the int8 network, compared with the reference from ONNX Runtime in
  `cpp/testdata/`.
- [`js/link.js`](js/link.js), [`js/channel.js`](js/channel.js), [`js/ofdm.js`](js/ofdm.js) and
  [`js/random.js`](js/random.js): the simulator of one slot. With the same bits and angles it gives the
  same signals as the Python simulator. Its own random numbers are checked by their statistics, because
  JavaScript cannot reproduce NumPy's generator.
- [`js/estimators.js`](js/estimators.js): LS + linear, the network input, equalization and bit errors.
  On one noisy slot it gives the same estimates and the same number of bit errors as Python.

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
