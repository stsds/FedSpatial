#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

// A bounded device-resident interval.  clock64 avoids data allocation and the
// duration is deliberately long enough for the three host intervals to show a
// common overlap, yet is independent of any Figure-1 / 15k code path.
__global__ void occupy_device(unsigned long long cycles) {
  const unsigned long long start = clock64();
  while (clock64() - start < cycles) {}
}

struct Interval {
  std::chrono::steady_clock::time_point start, end;
  float device_ms{};
};
}  // namespace

int main() {
  try {
    int device_count = 0;
    check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 3) throw std::runtime_error("requires CUDA devices 0,1,2");

    constexpr int kWorkers = 3;
    std::array<Interval, kWorkers> intervals;
    std::array<std::exception_ptr, kWorkers> errors;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::array<std::thread, kWorkers> workers;
    for (int worker = 0; worker < kWorkers; ++worker) {
      workers[worker] = std::thread([&, worker] {
        try {
          check(cudaSetDevice(worker), "cudaSetDevice");
          cudaDeviceProp properties{};
          check(cudaGetDeviceProperties(&properties, worker), "cudaGetDeviceProperties");
          cudaStream_t stream{};
          cudaEvent_t begin{}, finish{};
          check(cudaStreamCreate(&stream), "cudaStreamCreate");
          check(cudaEventCreate(&begin), "cudaEventCreate(begin)");
          check(cudaEventCreate(&finish), "cudaEventCreate(finish)");
          ready.fetch_add(1, std::memory_order_release);
          while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
          intervals[worker].start = std::chrono::steady_clock::now();
          check(cudaEventRecord(begin, stream), "cudaEventRecord(begin)");
          // clockRate is kHz, hence 250 ms * kHz is the requested cycle count.
          occupy_device<<<1, 1, 0, stream>>>(250ULL * properties.clockRate);
          check(cudaGetLastError(), "occupy_device launch");
          check(cudaEventRecord(finish, stream), "cudaEventRecord(finish)");
          check(cudaEventSynchronize(finish), "cudaEventSynchronize");
          intervals[worker].end = std::chrono::steady_clock::now();
          check(cudaEventElapsedTime(&intervals[worker].device_ms, begin, finish),
                "cudaEventElapsedTime");
          cudaEventDestroy(finish); cudaEventDestroy(begin); cudaStreamDestroy(stream);
        } catch (...) { errors[worker] = std::current_exception(); }
      });
    }
    while (ready.load(std::memory_order_acquire) != kWorkers) std::this_thread::yield();
    go.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);

    const auto latest_start = std::max({intervals[0].start, intervals[1].start,
                                        intervals[2].start});
    const auto earliest_end = std::min({intervals[0].end, intervals[1].end,
                                        intervals[2].end});
    const double common_overlap_ms = std::chrono::duration<double, std::milli>(
        earliest_end - latest_start).count();
    for (int worker = 0; worker < kWorkers; ++worker)
      std::cout << "CONCURRENCY_INTERVAL worker=" << worker << " device=" << worker
                << " device_ms=" << intervals[worker].device_ms << std::endl;
    std::cout << "CONCURRENCY_COMMON_OVERLAP_MS=" << common_overlap_ms << std::endl;
    if (common_overlap_ms < 100.)
      throw std::runtime_error("three-device activity did not overlap for 100 ms");
    std::cout << "CONCURRENCY_STATUS=PASS workers=3 synthetic_elements_per_worker=1 figure1_data=NO"
              << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CONCURRENCY_STATUS=FAIL reason=" << error.what() << std::endl;
    return 2;
  }
}
