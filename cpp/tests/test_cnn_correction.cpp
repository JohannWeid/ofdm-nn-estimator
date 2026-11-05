// C++ against the reference from Python, see scripts/make_test_vectors.py.

#include "cnn_correction.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::string kRoot = REPO_ROOT;
constexpr int kSlots = 5;
constexpr float kMaxDeviation = 1e-5F;

// Raw float32 in little endian, which is what x86-64 uses in memory.
std::vector<float> read_floats(const std::string& name) {
  const std::string path = kRoot + "/cpp/testdata/" + name + ".bin";
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("cannot open " + path);
  }
  const auto n_bytes = static_cast<std::size_t>(file.tellg());
  std::vector<float> values(n_bytes / sizeof(float));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(n_bytes));
  return values;
}

std::string model_path(const std::string& name) {
  return kRoot + "/models/" + name + ".onnx";
}

// The correction for every slot in input.bin, one slot at a time.
std::vector<float> run_all_slots(const std::string& model, int threads = 1) {
  const std::vector<float> input = read_floats("input");
  CnnCorrection network(model_path(model), threads);
  std::vector<float> correction(kSlots * CnnCorrection::kOutputSize);
  for (int slot = 0; slot < kSlots; ++slot) {
    network.run(input.data() + slot * CnnCorrection::kInputSize,
                correction.data() + slot * CnnCorrection::kOutputSize);
  }
  return correction;
}

float largest_difference(const std::vector<float>& a, const std::vector<float>& b) {
  float largest = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    largest = std::max(largest, std::abs(a[i] - b[i]));
  }
  return largest;
}

std::size_t count_different(const std::vector<float>& a, const std::vector<float>& b) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    count += a[i] != b[i] ? 1 : 0;
  }
  return count;
}

class ReferenceVectors : public ::testing::TestWithParam<std::string> {};

TEST_P(ReferenceVectors, HaveTheExpectedSize) {
  EXPECT_EQ(read_floats("input").size(), kSlots * CnnCorrection::kInputSize);
  EXPECT_EQ(read_floats(GetParam()).size(), kSlots * CnnCorrection::kOutputSize);
  EXPECT_EQ(read_floats(GetParam() + "_int8").size(), kSlots * CnnCorrection::kOutputSize);
}

TEST_P(ReferenceVectors, Fp32StaysWithinTheLimit) {
  const std::vector<float> expected = read_floats(GetParam());
  ASSERT_EQ(expected.size(), kSlots * CnnCorrection::kOutputSize);
  EXPECT_LT(largest_difference(run_all_slots(GetParam()), expected), kMaxDeviation);
}

TEST_P(ReferenceVectors, Int8IsBitIdentical) {
  const std::vector<float> expected = read_floats(GetParam() + "_int8");
  ASSERT_EQ(expected.size(), kSlots * CnnCorrection::kOutputSize);
  const std::vector<float> actual = run_all_slots(GetParam() + "_int8");
  EXPECT_EQ(count_different(actual, expected), 0U)
      << "largest difference " << largest_difference(actual, expected);
}

TEST_P(ReferenceVectors, Int8OnFourThreadsIsBitIdentical) {
  const std::vector<float> expected = read_floats(GetParam() + "_int8");
  ASSERT_EQ(expected.size(), kSlots * CnnCorrection::kOutputSize);
  EXPECT_EQ(count_different(run_all_slots(GetParam() + "_int8", 4), expected), 0U);
}

INSTANTIATE_TEST_SUITE_P(Models, ReferenceVectors,
                         ::testing::Values("cnn_A_log_nmse", "cnn_A_small"));

TEST(CnnCorrection, RunningASlotTwiceGivesTheSameResult) {
  const std::vector<float> input = read_floats("input");
  CnnCorrection network(model_path("cnn_A_log_nmse"));
  std::vector<float> first(CnnCorrection::kOutputSize);
  std::vector<float> second(CnnCorrection::kOutputSize);
  network.run(input.data(), first.data());
  network.run(input.data(), second.data());
  EXPECT_EQ(count_different(first, second), 0U);
}

TEST(CnnCorrection, LeavesTheInputUntouched) {
  const std::vector<float> input = read_floats("input");
  std::vector<float> copy = input;
  CnnCorrection network(model_path("cnn_A_log_nmse_int8"));
  std::vector<float> correction(CnnCorrection::kOutputSize);
  network.run(copy.data(), correction.data());
  EXPECT_EQ(count_different(copy, input), 0U);
}

TEST(CnnCorrection, MissingModelThrows) {
  EXPECT_THROW(CnnCorrection("does_not_exist.onnx"), Ort::Exception);
}

}  // namespace
