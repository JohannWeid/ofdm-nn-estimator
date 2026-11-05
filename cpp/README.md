# C++ inference (Phase 4)

`CnnCorrection` runs the correction network from `models/*.onnx` through the ONNX Runtime C++ API,
one slot at a time, FP32 or int8. `bench_latency` measures how long that takes.

It needs Linux on x86-64 (WSL works), a C++17 compiler and CMake 3.24 or newer. CMake fetches ONNX
Runtime and GoogleTest with fixed versions and checksums. In WSL, keep the build folder in the Linux
file system; access to `/mnt/c` is much slower. From the repository root:

```bash
cmake -S cpp -B ~/build/ofdm-cpp
cmake --build ~/build/ofdm-cpp --parallel
ctest --test-dir ~/build/ofdm-cpp --output-on-failure
```

## Latency

On a machine that is otherwise idle and on mains power, pinned to one virtual CPU. This writes
`results/latency.csv`; other output files, runs per repeat and repeats can be given as arguments:

```bash
taskset -c 2 ~/build/ofdm-cpp/bench_latency
```

`results/latency_under_load.csv` comes from the same command to that file, with a busy loop
(`while :; do :; done`) pinned to each of the other 11 virtual CPUs. `results/latency_later.csv` is
the same as `latency.csv`, from a later session.

## Test vectors

`testdata/` holds the reference for the tests, written by `python scripts/make_test_vectors.py`: raw
float32, little endian, in C order. `input.bin` has the shape (5, 3, 14, 72): one slot each at 0, 10,
20 and 30 dB SNR, then the first slot with doubled values, so that the int8 models have to clip their
input. `<model>.bin` has the shape (5, 2, 14, 72): the correction from `models/<model>.onnx` for each
slot, computed by ONNX Runtime in Python, one slot at a time on one thread.

The tests require less than 1e-5 deviation for FP32 and bit-identical values for int8.
