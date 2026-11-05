# Browser demo (Phase 5)

Planned: a static page on GitHub Pages with sliders for SNR, delay spread and Doppler, and
constellations for classical and neural estimation. No install, no backend, no packages.

So far, [`js/network.js`](js/network.js) runs the int8 network in plain JavaScript, without any
library. Its parameters come from `python scripts/export_web_data.py`, which writes `model/`. The
tests only need Node and compare the output with the reference from ONNX Runtime in `cpp/testdata/`.
From the repository root:

```bash
node --test "web/tests/*.test.mjs"
```
