// Latency of the correction network per slot, on one thread, for all four models.
//
// The models take turns, so that a slow phase of the machine does not hit only one of them. Run it
// on an otherwise idle machine and pin it to one core, for example with taskset -c 0.
//
// Usage: bench_latency [out.csv] [runs per repeat] [repeats]

#include "cnn_correction.hpp"
#include "test_vectors.hpp"

#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kRealSlots = 4;  // the fifth slot in input.bin is the artificial one with doubled values
constexpr int kWarmupRuns = 200;
constexpr double kSlotMs = 14 * (128 + 9) / 1920.0;  // 14 symbols of 137 samples at 1.92 MHz

struct Latency {
  double median_ms;
  double p95_ms;
  double p99_ms;
  double max_ms;
  long runs_over_slot;
};

// The value at or below which the given share of the sorted values lies.
double percentile(const std::vector<double>& sorted, double share) {
  const auto index = static_cast<std::size_t>(share * static_cast<double>(sorted.size() - 1));
  return sorted[index];
}

Latency measure(CnnCorrection& network, const std::vector<float>& input, int runs) {
  std::vector<float> correction(CnnCorrection::kOutputSize);
  const auto slot = [&input](int run) {
    return input.data() + (run % kRealSlots) * CnnCorrection::kInputSize;
  };
  for (int run = 0; run < kWarmupRuns; ++run) {
    network.run(slot(run), correction.data());
  }
  std::vector<double> ms(static_cast<std::size_t>(runs));
  for (int run = 0; run < runs; ++run) {
    const auto start = std::chrono::steady_clock::now();
    network.run(slot(run), correction.data());
    const auto stop = std::chrono::steady_clock::now();
    ms[static_cast<std::size_t>(run)] = std::chrono::duration<double, std::milli>(stop - start).count();
  }
  std::sort(ms.begin(), ms.end());
  const long over_slot = ms.end() - std::upper_bound(ms.begin(), ms.end(), kSlotMs);
  return {percentile(ms, 0.5), percentile(ms, 0.95), percentile(ms, 0.99), ms.back(), over_slot};
}

// The CPUs this process may run on, as "0" or "0 1 2".
std::string allowed_cpus() {
  cpu_set_t set;
  sched_getaffinity(0, sizeof(set), &set);
  std::string cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &set)) {
      cpus += (cpus.empty() ? "" : " ") + std::to_string(cpu);
    }
  }
  return cpus;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string out = argc > 1 ? argv[1] : std::string(REPO_ROOT) + "/results/latency.csv";
  const int runs = argc > 2 ? std::atoi(argv[2]) : 5000;
  const int repeats = argc > 3 ? std::atoi(argv[3]) : 3;

  const std::vector<std::string> names = {"cnn_A_log_nmse", "cnn_A_log_nmse_int8", "cnn_A_small",
                                          "cnn_A_small_int8"};
  const std::vector<float> input = read_test_vector("input");
  std::vector<std::unique_ptr<CnnCorrection>> networks;
  for (const std::string& name : names) {
    networks.push_back(
        std::make_unique<CnnCorrection>(std::string(REPO_ROOT) + "/models/" + name + ".onnx"));
  }

  const std::string cpus = allowed_cpus();
  std::ofstream csv(out);
  csv << "model,repeat,runs,median_ms,p95_ms,p99_ms,max_ms,runs_over_slot,cpus\n";
  std::printf("%-22s %6s %9s %9s %9s %9s %9s   cpus %s\n", "model", "repeat", "median ms", "p95 ms",
              "p99 ms", "max ms", "over slot", cpus.c_str());
  for (int repeat = 0; repeat < repeats; ++repeat) {
    for (std::size_t i = 0; i < names.size(); ++i) {
      const Latency l = measure(*networks[i], input, runs);
      std::printf("%-22s %6d %9.4f %9.4f %9.4f %9.4f %9ld\n", names[i].c_str(), repeat, l.median_ms,
                  l.p95_ms, l.p99_ms, l.max_ms, l.runs_over_slot);
      csv << names[i] << ',' << repeat << ',' << runs << ',' << l.median_ms << ',' << l.p95_ms << ','
          << l.p99_ms << ',' << l.max_ms << ',' << l.runs_over_slot << ',' << cpus << '\n';
    }
  }
  return 0;
}
