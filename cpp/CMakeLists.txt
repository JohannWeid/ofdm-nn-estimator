cmake_minimum_required(VERSION 3.24)
project(ofdm_cnn_correction LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"))
  message(FATAL_ERROR "The ONNX Runtime archive below is for Linux on x86-64; on Windows use WSL.")
endif()

include(FetchContent)

# The same version as the "deploy" extra in pyproject.toml.
FetchContent_Declare(
  onnxruntime
  URL https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-x64-1.31.1.tgz
  URL_HASH SHA256=c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9
)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.18.0/googletest-1.18.0.tar.gz
  URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(onnxruntime googletest)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so"
  INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
)

add_library(cnn_correction src/cnn_correction.cpp)
target_include_directories(cnn_correction PUBLIC include)
target_link_libraries(cnn_correction PUBLIC onnxruntime)
target_compile_options(cnn_correction PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
include(GoogleTest)
add_executable(test_cnn_correction tests/test_cnn_correction.cpp)
target_link_libraries(test_cnn_correction PRIVATE cnn_correction GTest::gtest_main)
target_compile_options(test_cnn_correction PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(test_cnn_correction PRIVATE tests)
target_compile_definitions(test_cnn_correction PRIVATE REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
gtest_discover_tests(test_cnn_correction)

add_executable(bench_latency bench/bench_latency.cpp)
target_link_libraries(bench_latency PRIVATE cnn_correction)
target_compile_options(bench_latency PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(bench_latency PRIVATE tests)
target_compile_definitions(bench_latency PRIVATE REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
# Only checks that the benchmark still runs; a real measurement is started by hand, see README.md.
add_test(NAME bench_latency_runs
         COMMAND bench_latency "${CMAKE_CURRENT_BINARY_DIR}/latency_smoke_test.csv" 10 1)
