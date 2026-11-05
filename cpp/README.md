# C++ inference (Phase 4)

Planned: ONNX Runtime C++ API, CMake build, GoogleTest against test vectors exported from
Python, and a latency benchmark (median and 95th percentile per slot).

`testdata/` holds those test vectors, written by `python scripts/make_test_vectors.py`: raw float32,
little endian, in C order. `input.bin` has the shape (5, 3, 14, 72): one slot each at 0, 10, 20 and
30 dB SNR, then the first slot with doubled values, so that the int8 models have to clip their
input. `<model>.bin` has the shape (5, 2, 14, 72): the correction from `models/<model>.onnx` for
each slot, computed by ONNX Runtime in Python, one slot at a time on one thread. The int8 outputs
are bit-identical on the x86 CPUs tested so far, with and without VNNI.
