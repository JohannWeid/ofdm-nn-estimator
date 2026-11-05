# Neural OFDM Channel Estimator with Embedded Deployment

> **Status: work in progress.** The CNN is trained and evaluated, its failure range is mapped
> and its building blocks are tested in ablations. It is exported to ONNX, quantized to int8 and
> runs in C++ within the slot time on a laptop. Try the
> [browser demo](https://johannweid.github.io/ofdm-nn-estimator/).

**Result so far:** On channels like its training data, a CNN with 5,378 parameters improves
the LS + linear channel estimate by 5.3 dB NMSE at 10 dB SNR, which is 1.2 dB SNR at a BER
of 1e-2 and 0.7 dB short of LMMSE with exact channel statistics. Its gain shrinks to 2.1 dB
at 30 dB SNR. On channels it has never seen (larger delay spread, higher Doppler) it clearly
helps only below 10 dB SNR and is up to 1.2 dB *worse* than LS + linear at 15-20 dB. A failure
map traces this to Doppler: the CNN smooths the channel over time by a nearly fixed amount,
whereas LMMSE adapts its smoothing to the channel. Ablations show that a tenth of the training
data nearly suffices, and that a model with 5.7 times fewer operations costs only 0.2 dB SNR
at that BER. Quantized to int8, the standard model loses at most 0.36 dB NMSE in region A and
0.02 dB SNR at that BER, with identical output on every x86 CPU type tested. In C++ it takes about
0.085 ms per slot (median) on one thread of a laptop, 9 % of the 1 ms slot, without the rest of the
receiver.

| Phase | Content | Status |
|---|---|---|
| 2 | Link-level simulation, LS / spline / LMMSE baselines | done |
| 3 | Neural estimator: data pipeline, model, training, evaluation, failure map, loss experiment, ablations | done |
| 4 | ONNX export, int8 quantization, C++ inference, latency, C++ tests in CI | done |
| 5 | Browser demo: page, simulator, receiver and int8 network in JavaScript | done |
| 5 | Browser demo: GitHub Pages | next |

## Method

A time-domain link-level simulator generates slots of 14 OFDM symbols x 72 subcarriers
(15 kHz spacing, 128-point FFT, 9-sample cyclic prefix). QPSK or 16-QAM data shares the
grid with a staggered pilot lattice (symbols 0/4/7/11, every 6th subcarrier). The channel
is a sample-spaced tapped delay line with an exponential power delay profile and
per-tap Rayleigh fading from a sum-of-sinusoids (Jakes) model, followed by AWGN. The
receiver removes the CP, applies the FFT, estimates the channel, and equalizes with one tap
per resource element.

Channel conditions come in two regions, so that robustness claims can be checked:

| Region | Delay spread | Max. Doppler | Use |
|---|---|---|---|
| A | 0.1-0.6 us | 0-150 Hz | training + in-distribution test |
| B | 0.9-1.3 us | 300-600 Hz | out-of-distribution test |

### Neural estimator

The network refines the LS + linear estimate instead of estimating the channel from
scratch:

- **Input** (3 x 14 x 72): real and imaginary part of the LS + linear estimate, plus the
  SNR as a constant plane (`snr_db / 30`). The same function builds the input for
  training and inference ([`src/nn/dataset.py`](src/nn/dataset.py)).
- **Model** ([`src/nn/model.py`](src/nn/model.py)): four 3x3 convolutions
  (3-16-16-16-2 channels, ReLU) whose output is *added* to the LS + linear estimate. The
  last layer starts at zero, so the untrained model reproduces the baseline exactly.
  It has 9 x (3x16 + 16x16 + 16x16 + 16x2) = 5,328 weights plus 50 biases, so 5,378
  parameters, and needs 5,328 x 14 x 72 = 5.37 M multiply-accumulates per slot.
- **Data**: 32,000 training slots from region A with SNR uniform in 0-30 dB, generated
  in memory from fixed seeds (under 2 minutes on a laptop CPU) instead of being stored.
  Validation and test sets use separate seeds; region B is never used for training.
- **Training**: Adam with a log-NMSE loss, the mean NMSE in dB over the slots, so that every
  slot counts equally. It used the full 100 epochs, about 33 minutes on a laptop CPU
  ([training curve](results/training_A_log_nmse.png)). The best epoch is the last one, but the
  last 20 epochs only gained 0.04 dB. Validation NMSE over mixed SNR: -15.7 dB, against -9.8 dB
  for the network input.

Note on fairness: the network is given the exact SNR, the interpolation baselines are not.
[Without the SNR plane](#what-each-building-block-contributes) its gain in region A falls from
5.3 to 4.7 dB at 10 dB SNR and disappears at 30 dB.

## Reproduce

```bash
pip install -e ".[dev]"
pytest
python scripts/run_baselines.py --region A
```

The model tests and the CNN results need PyTorch (CPU is enough); the tests are skipped
without it. With PyTorch installed, the trained model in `models/` can be evaluated directly:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/evaluate_nn.py --region A
```

`python scripts/train_nn.py` repeats the training with the same data and start weights. On the
laptop used here, a fresh process matched the logged values of the first two epochs exactly;
other machines or thread counts can differ slightly. Adding
`--nn-config configs/experiments/<name>.yaml` trains a variant.

The ONNX export and the int8 quantization need one more set of packages. The comparison takes
roughly half an hour on a laptop CPU:

```bash
pip install -e ".[deploy]"
python scripts/export_onnx.py
python scripts/compare_quantization.py
```

The C++ inference needs Linux on x86-64 (WSL works), see [`cpp/README.md`](cpp/README.md):

```bash
cmake -S cpp -B ~/build/ofdm-cpp
cmake --build ~/build/ofdm-cpp --parallel
ctest --test-dir ~/build/ofdm-cpp --output-on-failure
```

All parameters live in [`configs/default.yaml`](configs/default.yaml); every figure is
produced by a script from a fixed seed.

## Results

QPSK, 4000 independent test slots per SNR point, `python scripts/evaluate_nn.py --region A|B`.
The CNN was trained on region A only.

![CNN and baselines in region A](results/nn_eval_A.png)
![CNN and baselines in region B](results/nn_eval_B.png)

| NMSE (dB) | A @ 0 dB | A @ 10 dB | A @ 20 dB | A @ 30 dB | B @ 0 dB | B @ 10 dB | B @ 20 dB | B @ 30 dB |
|---|---|---|---|---|---|---|---|---|
| LS + linear | -1.6 | -11.5 | -21.3 | -29.2 | -1.4 | -10.2 | -14.9 | -15.9 |
| LS + spline | -0.1 | -10.1 | -20.0 | -28.9 | 0.0 | -9.2 | -15.2 | -16.7 |
| **CNN** | **-7.9** | **-16.9** | **-24.6** | **-31.3** | **-5.6** | **-10.0** | **-14.1** | **-16.2** |
| LMMSE (known statistics) | -12.4 | -20.2 | -28.6 | -37.2 | -7.5 | -14.4 | -21.8 | -26.5 |

| SNR needed for the target BER | LS + linear | CNN | LMMSE | Perfect CSI |
|---|---|---|---|---|
| Region A, BER 1e-2 | 19.4 dB | 18.2 dB | 17.5 dB | 16.9 dB |
| Region A, BER 3e-2 | 14.1 dB | 12.7 dB | 12.2 dB | 11.7 dB |
| Region B, BER 3e-2 | 16.0 dB | 17.6 dB | 13.5 dB | 11.9 dB |

**Region A (same conditions as in training).** The CNN gains 6.3 dB NMSE over LS + linear at
0 dB SNR, without being told the channel statistics. The gain then shrinks steadily to 2.1 dB
at 30 dB, while the distance to LMMSE grows from 3.3 dB at 10 dB to 5.8 dB at 30 dB.

**Region B (never seen in training).** The CNN still helps at low SNR (+4.2 dB at 0 dB) but is
worse than its own input at 15 and 20 dB, by up to 1.2 dB. At 10 and 25 dB it is about level,
within the noise between training runs, and at 30 dB it is 0.4 dB better. At a BER of 3e-2 this
costs 1.6 dB SNR compared to plain LS + linear; the section on failures below explains why.
**The model should not be used outside its training range without a check.**

**Classical baselines.** Both LS variants hit an error floor around -16 dB in region B: the
channel changes too much between pilot symbols for interpolation. LMMSE avoids the floor
because it knows the Doppler, but it needs exact statistics. Even perfect CSI shows a BER
floor in region B (1.4e-3 at 30 dB vs. 6e-4 in A), caused by inter-carrier interference and
delay taps beyond the cyclic prefix. Baselines alone: `scripts/run_baselines.py`.

### Does the training loss matter?

The first model used an unweighted MSE ([`configs/experiments/mse.yaml`](configs/experiments/mse.yaml))
and gained only 0.1 dB at 30 dB SNR. Its error energy is 127 times larger at 0 dB than at 30 dB,
so noisy slots dominate that loss. The log-NMSE loss tests whether this causes the weak gain.
Data and start weights are the same. All other results in this README use the log-NMSE model,
because it gains more on average over the SNR points (4.3 dB against 3.8 dB in region A).

![CNN with MSE and log-NMSE loss in region A](results/loss_comparison_A.png)

| NMSE gain over LS + linear (dB) | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| Region A, MSE loss | 6.7 | 6.2 | 5.4 | 4.2 | 2.6 | 1.3 | 0.1 |
| Region A, log-NMSE loss | 6.3 | 6.0 | 5.3 | 4.4 | 3.3 | 2.6 | 2.1 |
| Region B, MSE loss | 4.3 | 1.9 | -0.4 | -1.8 | -1.6 | -0.7 | -0.1 |
| Region B, log-NMSE loss | 4.2 | 2.1 | -0.2 | -1.2 | -0.8 | -0.1 | 0.4 |

- **The loss is part of the explanation, not all of it.** At 30 dB in region A the gain rises from
  0.1 to 2.1 dB, for 0.4 dB less at 0 dB. The CNN is still 5.8 dB behind LMMSE there.
- **It does not fix region B.** The worst value improves from -1.8 to -1.2 dB, but the CNN stays
  worse than LS + linear at 15 and 20 dB. This fits the Doppler range being the main problem
  rather than the weighting of the slots.
- **Caveat:** one training run per loss, both over the full 100 epochs (best epochs 99 and 100).

### Where the CNN fails

`python scripts/failure_map.py`, 300 slots per cell and SNR. All estimators see the same slots.
Blue means the CNN is better than the reference, red means it is worse.

![NMSE difference of the CNN over delay spread and Doppler](results/failure_map.png)

- **Doppler does the damage.** Without Doppler the CNN beats LS + linear at every delay spread
  and SNR in the map. A larger delay spread only shrinks its advantage: at 10 dB SNR the gain
  falls from 5.9 dB at 0.1 us to 1.0 dB at 1.5 us.
- **Where it turns harmful depends on the SNR.** Training only covered up to 150 Hz. The CNN
  falls behind LS + linear from 400-600 Hz at 10 dB SNR and from 300-500 Hz at 20 dB. At 30 dB
  the range is widest: from 200 Hz at 0.1 us but only from 800 Hz at 1.5 us. The largest loss in
  the map is 2.9 dB (20 dB SNR, 0.1 us, 600 Hz).
- **Inside its training range it is reliably better:** between 1.3 and 3.0 dB at 30 dB SNR, and
  more at lower SNR.

To check whether smoothing explains this, `python scripts/tracking_check.py` measures which
share of the true channel change over 3 OFDM symbols arrives in the estimate (1 = all of it;
noise does not bias this measure). Delay spread 0.3 us:

| Share of true change over time | 10 dB, 50 Hz | 10 dB, 600 Hz | 20 dB, 50 Hz | 20 dB, 600 Hz |
|---|---|---|---|---|
| LS + linear | 1.00 | 0.93 | 0.99 | 0.93 |
| CNN | 0.63 | 0.54 | 0.80 | 0.72 |
| LMMSE (known statistics) | 0.56 | 0.95 | 0.90 | 0.99 |

Smoothing itself is not the problem: at low Doppler LMMSE smooths at least as much at 10 dB
SNR. But LMMSE knows the Doppler and stops smoothing when the channel changes quickly, while
the CNN keeps smoothing at nearly the same strength. This fits the failure map. That the
fixed smoothing comes from training only on low Doppler is still an untested assumption;
training over a wider Doppler range would test it.

### What each building block contributes

`python scripts/compare_ablations.py`, on the same test slots as above. Each variant changes
one thing in the standard model and was trained once. The data variants get proportionally
more epochs, so that less data does not also mean less training.

| NMSE gain over LS + linear (dB) | A, 10 dB | A, 20 dB | A, 30 dB | B, 10 dB | B, 20 dB | B, 30 dB | SNR for BER 1e-2 in A |
|---|---|---|---|---|---|---|---|
| **Standard** (5,378 parameters, 32,000 slots) | **5.3** | **3.3** | **2.1** | **-0.2** | **-0.8** | **0.4** | **18.2 dB** |
| Other seed | 5.3 | 3.3 | 2.1 | 0.0 | -1.0 | 0.4 | 18.2 dB |
| No SNR plane | 4.7 | 3.1 | -0.1 | -0.5 | -4.2 | -5.0 | 18.2 dB |
| Small model (8 channels, 3 layers, 954 parameters) | 4.1 | 2.6 | 0.8 | 0.5 | -1.5 | -0.4 | 18.4 dB |
| 30 % of the data (9,600 slots) | 5.3 | 3.3 | 2.0 | -0.2 | -1.1 | 0.0 | 18.2 dB |
| 10 % of the data (3,200 slots) | 5.2 | 3.2 | 1.9 | -0.2 | -0.8 | 0.3 | 18.2 dB |

For comparison, LS + linear needs 19.4 dB SNR for a BER of 1e-2 in region A.

- **Noise between training runs sets the scale.** Over all seven SNR points the second seed
  differs from the standard model by at most 0.06 dB in region A and 0.14 dB in region B. Region B
  varies more than that suggests: an earlier run that stopped after 71 epochs was up to 0.6 dB
  lower at 20 dB, and from 15 dB on the 30 % model is worse than both the 10 % and the full
  model. Differences below about 0.5 dB in region B should not be read as real.
- **The SNR plane matters most at high SNR.** Without it the gain in region A disappears at
  30 dB, and in region B the model ends up 5.0 dB worse than LS + linear. At a BER of 1e-2 it
  costs only 0.06 dB SNR, so the main result does not depend on this input. The plane lets the
  network smooth less when the noise is low: at 600 Hz and 0.3 us, the share of the true change
  over time rises from 0.54 at 10 dB to 0.72 at 20 dB with the plane, but only from 0.52 to 0.54
  without it (`python scripts/tracking_check.py --model models/cnn_A_no_snr.pt --out results/no_snr`).
- **A much smaller model costs little at link level.** The small model needs 5.7 times fewer
  multiply-accumulates (0.94 M per slot) and loses 0.7-1.5 dB NMSE in region A, but only 0.2 dB
  SNR at a BER of 1e-2, because noise dominates the channel estimation error at that SNR. In
  region B it is neither clearly better nor worse: +0.7 dB at 10 dB, -0.7 to -1.0 dB from 20 dB
  on. In int8 it loses more of its gain, see [below](#int8-quantization).
- **Data is not the bottleneck.** With 10 % of the slots the gain in region A falls by 0.25 dB at
  30 dB and at most 0.13 dB elsewhere. More slots from region A are therefore unlikely to close
  the gap to LMMSE; whether a larger model would is untested. Early stopping ended the 10 % run
  after 849 epochs, so it got 21,225 gradient steps instead of 25,000.

### ONNX export

`python scripts/export_onnx.py` exports only the correction network; adding the correction to the
LS + linear estimate stays outside the model. That way the int8 quantization does not round the
estimate itself. The largest output difference to PyTorch over 100 validation slots is 1.4e-6 for
the standard model and 8.3e-7 for the small one. Evaluated in ONNX Runtime on the test slots above,
the standard model's NMSE differs by less than 1e-6 dB and its bit errors by at most 1 in 7.68
million bits per SNR point.

### int8 quantization

The same script quantizes both models with ONNX Runtime after training (static quantization, QDQ
format, no retraining), calibrated on 2,000 validation slots. `python scripts/compare_quantization.py`
evaluates them on the test slots above. Only the correction is int8: adding it to the LS + linear
estimate stays in floating point, so a purely integer receiver would need a wider format for that
step.

| Region A: NMSE in dB | 0 dB | 10 dB | 20 dB | 30 dB | Largest NMSE loss | SNR for BER 1e-2 in A | SNR for BER 3e-2 in B | File |
|---|---|---|---|---|---|---|---|---|
| LS + linear | -1.57 | -11.53 | -21.25 | -29.20 | | 19.36 dB | 16.01 dB | |
| Standard, FP32 | -7.87 | -16.86 | -24.56 | -31.34 | | 18.16 dB | 17.63 dB | 30.2 KB |
| **Standard, int8** | **-7.51** | **-16.80** | **-24.46** | **-30.99** | **0.36 dB** | **18.18 dB** | **17.66 dB** | **16.5 KB** |
| Standard, int8, 8-bit weights | -7.56 | -16.84 | -24.48 | -31.02 | 0.33 dB | 18.17 dB | 17.65 dB | |
| Standard, int8, full value range | -7.83 | -16.73 | -24.16 | -30.20 | 1.15 dB | 18.24 dB | 17.85 dB | |
| Standard, int8, naive | -7.82 | -16.72 | -23.92 | -29.27 | 2.07 dB | 18.30 dB | 17.94 dB | |
| Small, FP32 | -6.37 | -15.63 | -23.82 | -30.03 | | 18.37 dB | 18.15 dB | 10.6 KB |
| Small, int8 | -6.12 | -15.44 | -23.21 | -28.31 | 1.72 dB | 18.50 dB | 18.40 dB | 9.3 KB |

The largest loss is over all seven SNR points of region A, against the FP32 version of the same
model. All int8 rows use 7-bit weights, except "8-bit weights" and "naive". The naive variant has
the addition inside the model, one weight step size per layer and the full value range. Rows with
8-bit weights hold for a CPU with VNNI, see below.

- **Done carefully, int8 costs little.** The standard model keeps 1.80 of its 2.15 dB gain over
  LS + linear at 30 dB. In region B it loses at most 0.15 dB, within the noise between training
  runs. All convolutions run as `QLinearConv`, so in integers; the tests check this on the graph
  ONNX Runtime really executes, because the file alone does not show it.
- **With 8-bit weights the result depends on the CPU.** The CI found this: the same file gave
  -29.88 dB on its runner where the laptop (with AVX-VNNI) gave -30.92 dB, on 64 slots at 30 dB.
  x86 CPUs without VNNI add pairs of products in 16 bits, where 8-bit weights can overflow. CPUs
  emulated with QEMU (Haswell, EPYC-Milan) reproduced the CI numbers exactly and changed 23 % of the
  outputs. With 7-bit weights, the outputs are bit-identical on Windows, Linux, both emulated CPUs
  and the CI runners seen so far (AMD EPYC 7763 without VNNI, Intel Xeon 6975P-C and AMD EPYC 9V74
  with AVX-512 VNNI), and the standard model loses at most 0.04 dB against 8 bits. The runners change from job to job,
  so the CI also runs the C++ tests on an emulated Haswell. The comparison of 8-bit and 7-bit files
  on emulated CPUs was made by hand and is not in a script. Other architectures were not tested.
- **The naive way loses the gain at high SNR.** Only 0.07 dB remain at 30 dB. Moving the addition
  out and giving every filter its own step size brings back 0.93 dB, limiting the value range
  another 0.79 dB. In the naive model ONNX Runtime also leaves one convolution in floating point
  without a warning; the cause was not investigated.
- **The step size shows why.** At 30 dB the LS + linear error is about 0.025 per real or imaginary
  part. One step at the output is 0.026 with the full value range and 0.013 with the limited one.
  Rounding adds step^2 / 12 per part, which by calculation explains about half of the remaining
  0.35 dB at 30 dB. The rest has to come from the rounding inside the network.
- **The range is a compromise, chosen on validation slots.** Clipping rare values costs accuracy at
  low SNR, where they occur: 0.36 dB at 0 dB, against 0.05 dB with the full range. Of five ranges
  ([`results/quantization_percentile.csv`](results/quantization_percentile.csv)), covering 99.95 %
  of the values gives the smallest largest loss on validation slots: 0.36 dB, against 0.55 dB for
  99.9 % and 0.53 dB for 99.995 %. The neighbouring 99.98 % is only 0.05 dB behind, so the exact
  value should not be overread.
- **The small model cannot afford this.** It loses up to 1.72 dB and ends up 0.89 dB *worse* than
  LS + linear at 30 dB. At a BER of 1e-2 it falls 0.32 dB behind the standard model, against 0.21 dB
  in FP32. In int8 it is also [barely faster](#latency), so it is no replacement for the standard
  model.

### C++ inference

[`cpp/`](cpp/) runs both models through the ONNX Runtime C++ API, one slot at a time, on the memory of
the caller. GoogleTest compares the result with reference outputs from ONNX Runtime in Python
(`python scripts/make_test_vectors.py`): five slots, one of which forces the int8 models to clip
their input. All 10,080 values per model are bit-identical, for FP32 and int8, also on the emulated
Haswell and EPYC-Milan CPUs. A build with AddressSanitizer and UndefinedBehaviorSanitizer runs clean.

Both sides use the same library, so this checks the C++ wrapper and the data layout, not an
independent implementation. [`web/js/network.js`](web/js/network.js) is one: plain JavaScript without
any library, which does the rounding, zero points and scaling itself. Its int8 output matches the same
reference bit for bit, and also ONNX Runtime on 600 further slots with values far beyond the int8
range (once, by hand). It takes about 6 ms per slot in Node, so it is a check and a demo, not a
replacement for the C++ code. The whole chain in JavaScript (simulator, LS + linear, network) stays
within 0.19 dB of the NMSE values above in both regions, with its own random numbers and 1000 slots
per SNR (`node web/tests/compare_with_python.mjs`). Runs of that length differ by about 0.1 dB
between seeds, in Python as well.

The sanitizer run was made by hand. The CI builds the C++ code and runs
its tests on every push, on the CPU of the runner and again on an emulated Haswell without VNNI.

### Latency

`taskset -c 2 ~/build/ofdm-cpp/bench_latency`, see [`cpp/README.md`](cpp/README.md): one thread, three
repeats of 5,000 runs per model, for the network alone. A slot lasts 0.999 ms. Measured in WSL 2, so
in a Linux VM, on a laptop with an i7-1355U on mains power. Ranges show the three repeats.

| Per slot | Median | p95 | p99 | Largest | Runs over the slot (of 15,000) | All other CPUs busy: median | p99 | Runs over the slot |
|---|---|---|---|---|---|---|---|---|
| Standard, FP32 | 0.131 ms | 0.17 ms | 0.32-0.35 ms | 4.0 ms | 8 | 0.34 ms | 0.67-0.84 ms | 92 |
| **Standard, int8** | **0.081-0.085 ms** | **0.10-0.11 ms** | **0.23-0.27 ms** | **3.5 ms** | **10** | **0.27 ms** | **0.51-0.63 ms** | **58** |
| Small, FP32 | 0.041 ms | 0.05 ms | 0.10-0.13 ms | 0.58 ms | 0 | 0.07-0.12 ms | 0.20-0.32 ms | 21 |
| Small, int8 | 0.063 ms | 0.08 ms | 0.16-0.18 ms | 1.4 ms | 4 | 0.16 ms | 0.35-0.77 ms | 65 |

- **Well inside the slot on this laptop.** The int8 standard model needs 9 % of the slot in the
  median, and with all other CPUs busy 27 %. FFT, interpolation, equalization and decoding share the
  same slot and are not included.
- **Not real-time.** Single runs exceed the slot, up to 3.5 ms for the int8 standard model, in 10 of
  15,000 runs. A VM on Windows gives no timing guarantee, so this tail says more about the system
  than about the network. A real-time system was not tested.
- **int8 does not simply mean faster.** The standard model gains a factor of 1.5 against FP32, the
  small one loses that much. The profiler of ONNX Runtime shows why (`python scripts/latency_python.py`,
  [`results/latency_layers_linux.csv`](results/latency_layers_linux.csv)): an int8 layer takes 21-31 us
  almost regardless of its size, and 16 to 16 channels is even faster than 8 to 8. In FP32 the time
  grows with the size. So 5.7 times fewer operations save only about a quarter in int8. The weights
  are not the cause: in a run by hand, 7 bits, 8 bits and one step size per layer took the same time.
  What causes this floor inside ONNX Runtime was not investigated.
- **The platform matters as much as the model.** In the same script, the int8 file takes 0.150 ms on
  Windows on the same laptop and 0.096 ms in WSL, FP32 takes 0.163 and 0.156 ms. The reason was not
  investigated.
- **Sessions differ.** A second C++ session on the same day ([`results/latency_later.csv`](results/latency_later.csv))
  was 7-16 % slower in the median, and one single run took 34 ms. The reason was not investigated.
  Differences of that size between sessions should not be read as real.
- **Which core ran is unknown.** The i7-1355U has 2 fast and 8 slow cores, but WSL hides them: all 12
  virtual CPUs give the same speed. Presumably Windows puts a single busy thread on a fast core. The
  run under load shows what happens without one.

### From PyTorch to C++

| Region A | File | Median per slot | p99 | NMSE at 30 dB | Largest NMSE loss | SNR for BER 1e-2 |
|---|---|---|---|---|---|---|
| FP32, PyTorch, Python on Windows | 24.8 KB | 0.222 ms | 0.70-0.84 ms | -31.34 dB | | 18.16 dB |
| FP32, ONNX Runtime, Python on Windows | 30.2 KB | 0.163 ms | 0.38-0.45 ms | -31.34 dB | | 18.16 dB |
| FP32, ONNX Runtime, C++ in WSL | 30.2 KB | 0.131 ms | 0.32-0.35 ms | -31.34 dB | | 18.16 dB |
| **int8, ONNX Runtime, C++ in WSL** | **16.5 KB** | **0.081-0.085 ms** | **0.23-0.27 ms** | **-30.99 dB** | **0.36 dB** | **18.18 dB** |

Standard model, one thread, network only. The two Python rows come from the later session, in which
C++ was also slower, so read them against each other and not against the C++ rows.

### What would change on embedded hardware

All of this ran on a laptop, not on a DSP, an NPU or a baseband chip, and there are no measurements
for such a target here. What can be said without one:

- **Computing power.** The standard model needs 5.37 M multiply-accumulates per slot, so a target has
  to sustain 5.4 G per second just to fill the whole slot with the network, and clearly more because
  FFT, equalization and decoding share it. The small model needs 0.94 G per second, at the accuracy
  cost shown above.
- **Memory.** The int8 weights take 5,328 bytes, the 50 int32 biases 200 bytes. The largest activation
  is 16 x 14 x 72 = 16,128 bytes, and a convolution needs its input and output at the same time, so
  about 32 KB without further tricks, a quarter of FP32. The runtime is the bigger problem: ONNX
  Runtime unpacks to 30 MB for this 16.5 KB model. A small target would need its own kernels.
- **No floating point unit.** Only the correction is int8 here. The LS + linear estimate, the addition
  and the conversion of the network's input and output are floating point, and they would need a
  fixed-point format, probably wider than 8 bits.
- **Other kernels, other results.** The CPU dependence of 8-bit weights and the floor per layer come
  from the x86 kernels of ONNX Runtime. Another instruction set can saturate or round differently, so
  accuracy has to be measured again on the target, not only speed.
- **Timing.** Median and p99 say little about a hard deadline. That needs a real-time system and a
  worst-case analysis, neither of which is part of this project.

### Limitations

- LMMSE is not a practical competitor here but an upper bound: it knows the exact per-slot
  PDP, Doppler and noise variance. A fair classical opponent with averaged statistics is
  still missing.
- The CNN results are for QPSK. Every variant was trained once; only the standard model has a
  second seed.
- The network gets the exact SNR. A real receiver has to estimate it, and the effect of an SNR
  error is not tested.
- Taps are sample-spaced (Ts = 0.52 us), so small delay spreads are coarsely resolved.
- NMSE is computed over all resource elements, including the pilot positions.
- Channels with a delay spread beyond the CP cause ISI that no estimator here can remove.
- Both interpolation baselines extrapolate linearly beyond the outermost pilots. Cubic
  extrapolation amplified the pilot noise by up to 18 dB at the grid edges.

## Repository layout

```
configs/            YAML configurations, variants in configs/experiments/
src/ofdm/           simulation, channel, baselines
src/nn/             training data, CNN model, training, estimator wrapper, export, quantization
models/             trained weights, ONNX exports and their int8 versions
tests/              pytest
cpp/                C++ inference, CMake, GoogleTest, test vectors in testdata/
web/                browser demo: page, simulator, receiver and int8 network in JavaScript
scripts/            entry points that produce results/
results/            plots and tables
.github/workflows/  CI
```

## License

MIT
