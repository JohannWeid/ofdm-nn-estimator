# C++ inference (Phase 4)

`CnnCorrection` runs the correction network from `models/*.onnx` through the ONNX Runtime C++ API,
one slot at a time, FP32 or int8. A latency benchmark is still planned.

It needs Linux on x86-64 (WSL works), a C++17 compiler, CMake 3.24 or newer and Ninja. CMake fetches
ONNX Runtime and GoogleTest with fixed versions and checksums. In WSL, keep the build folder in the
Linux file system; access to `/mnt/c` is much slower. From the repository root:

```bash
cmake -S cpp -B ~/build/ofdm-cpp -G Ninja
cmake --build ~/build/ofdm-cpp
ctest --test-dir ~/build/ofdm-cpp --output-on-failure
```

## Test vectors

`testdata/` holds the reference for the tests, written by `python scripts/make_test_vectors.py`: raw
float32, little endian, in C order. `input.bin` has the shape (5, 3, 14, 72): one slot each at 0, 10,
20 and 30 dB SNR, then the first slot with doubled values, so that the int8 models have to clip their
input. `<model>.bin` has the shape (5, 2, 14, 72): the correction from `models/<model>.onnx` for each
slot, computed by ONNX Runtime in Python, one slot at a time on one thread.

The tests require less than 1e-5 deviation for FP32 and bit-identical values for int8.
