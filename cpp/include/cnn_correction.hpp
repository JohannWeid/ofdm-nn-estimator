#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>

// Runs the correction network of the neural channel estimator on one slot.
//
// Input: real and imaginary part of the LS + linear estimate and the SNR plane (snr_db / 30), as
// built by nn.dataset.to_input in Python. Output: the correction, which the caller adds to the
// real and imaginary part of the estimate. The same class runs the FP32 and the int8 model.
class CnnCorrection {
 public:
  static constexpr int kSymbols = 14;
  static constexpr int kSubcarriers = 72;
  static constexpr int kInputSize = 3 * kSymbols * kSubcarriers;
  static constexpr int kOutputSize = 2 * kSymbols * kSubcarriers;

  // Throws Ort::Exception if the model cannot be loaded.
  explicit CnnCorrection(const std::string& model_path, int threads = 1);

  // input holds kInputSize values and correction has room for kOutputSize values, both in the
  // order channel, symbol, subcarrier. Throws Ort::Exception if the model does not fit these sizes.
  void run(const float* input, float* correction);

 private:
  Ort::Env env_;
  Ort::Session session_;
  Ort::MemoryInfo memory_;
  std::string input_name_;
  std::string output_name_;
};
