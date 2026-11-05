#include "cnn_correction.hpp"

#include <array>
#include <cstdint>

namespace {

Ort::SessionOptions make_options(int threads) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(threads);
  options.SetInterOpNumThreads(1);
  // Otherwise idle worker threads keep spinning and a latency measurement is not per core.
  options.AddConfigEntry("session.intra_op.allow_spinning", "0");
  return options;
}

}  // namespace

CnnCorrection::CnnCorrection(const std::string& model_path, int threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "cnn_correction"),
      session_(env_, model_path.c_str(), make_options(threads)),
      memory_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {
  Ort::AllocatorWithDefaultOptions allocator;
  input_name_ = session_.GetInputNameAllocated(0, allocator).get();
  output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
}

void CnnCorrection::run(const float* input, float* correction) {
  static constexpr std::array<std::int64_t, 4> input_shape{1, 3, kSymbols, kSubcarriers};
  static constexpr std::array<std::int64_t, 4> output_shape{1, 2, kSymbols, kSubcarriers};

  // Both tensors only wrap the memory of the caller, nothing is copied. ONNX Runtime does not
  // write to an input, its interface just does not take a pointer to const.
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_, const_cast<float*>(input), kInputSize, input_shape.data(), input_shape.size());
  Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
      memory_, correction, kOutputSize, output_shape.data(), output_shape.size());

  const char* input_name = input_name_.c_str();
  const char* output_name = output_name_.c_str();
  session_.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name,
               &output_tensor, 1);
}
