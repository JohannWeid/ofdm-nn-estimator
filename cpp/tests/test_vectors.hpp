#pragma once

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Reads cpp/testdata/<name>.bin: raw float32 in little endian, which is what x86-64 uses in memory.
inline std::vector<float> read_test_vector(const std::string& name) {
  const std::string path = std::string(REPO_ROOT) + "/cpp/testdata/" + name + ".bin";
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
