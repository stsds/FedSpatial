#include "fedspatial/nworker_15k_figure1_config.hpp"

#include <cuda.h>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fedspatial;

static WorkerInput worker(int id) {
  Matrix sites(3, 2), z(3, 1), x(3, NWorker15kFigure1Config::covariates);
  for (int row = 0; row < 3; ++row) {
    sites(row, 0) = .1 * (id * 3 + row); sites(row, 1) = .2 * (id + row);
    z(row, 0) = .25 * (row + 1);
    // The compact gamma solve is q=5.  Across the two three-row workers use
    // a full-rank Vandermonde design, rather than the former shifted columns
    // (which had rank two and made the deliberately small 5-by-5 normal
    // matrix singular before the streamed path could be exercised).
    const double t = double(id * 3 + row) - 2.5;
    for (int col = 0; col < x.cols; ++col) {
      x(row, col) = 1.;
      for (int power = 0; power < col; ++power) x(row, col) *= t;
    }
  }
  return {sites, z, x};
}

static size_t cuda_allocation_bytes(const double* pointer) {
  if (!pointer) return 0;
  CUdeviceptr base = 0;
  size_t bytes = 0;
  const auto status = cuMemGetAddressRange(
      &base, &bytes, reinterpret_cast<CUdeviceptr>(pointer));
  if (status != CUDA_SUCCESS)
    throw std::runtime_error("cuMemGetAddressRange failed");
  return bytes;
}

int main() {
  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) return 77;
    Matrix knots(2, 2); knots(0, 0) = knots(0, 1) = 0.; knots(1, 0) = knots(1, 1) = 1.;
    auto p = nworker_15k_figure1_initial(2);
    std::vector<WorkerInput> workers{worker(0), worker(1)};
    NWorkerPSynchronousExecutor executor({0, 0});
    (void)executor.run_round_with_checkpoints(workers, knots, p);
    bool valid = true;
    for (size_t worker_id = 0; worker_id < executor.contexts_.size(); ++worker_id) {
      const auto& context = executor.contexts_[worker_id];
      const auto& region = context->region();
      const auto b = context->b_derivative_allocation_count();
      const auto a = context->a_derivative_allocation_count();
      const auto m = context->m_derivative_allocation_count();
      const auto pair = context->b_pair_derivative_allocation_count();
      const auto reduction = context->delta_theta_reduction_allocation_count();
      const auto global_k = context->global_k_correction_allocation_count();
      const size_t nn = size_t(region.n) * region.n;
      const size_t nm = size_t(region.n) * region.m;
      const size_t mm = size_t(region.m) * region.m;
      const size_t forbidden_bytes = 25 * nn * sizeof(double);
      const size_t pair_bytes = 3 * (nm + mm) * sizeof(double);
      const size_t final_bytes = region.delta_theta_final_capacity * sizeof(double);
      const size_t component_bytes = region.delta_theta_component_scratch_capacity * sizeof(double);
      const size_t global_k_bytes = region.delta_theta_reduction_scratch_capacity * sizeof(double);
      const size_t final_range_bytes = cuda_allocation_bytes(region.delta_theta_final);
      const size_t summary_range_bytes = cuda_allocation_bytes(region.delta_theta_summary);
      const size_t component_range_bytes = cuda_allocation_bytes(region.delta_theta_component_scratch);
      const size_t global_k_range_bytes = cuda_allocation_bytes(region.delta_theta_reduction_scratch);
      const size_t expected_global_k = worker_id == 0 ? 10 * mm * sizeof(double) : 0;
      const size_t global_k_derivative_bytes = worker_id == 0 ? 5 * mm * sizeof(double) : 0;
      const long long expected_reductions = worker_id == 0 ? 2 : 1;
      const bool no_legacy_bundle = b == 0 && a == 0 && m == 0 &&
          !region.b_alpha && !region.a_alpha && !region.m_delta;
      const bool compact_pair = pair == 1 && region.b_pair_nm_capacity == nm &&
          region.b_pair_mm_capacity == mm && pair_bytes < forbidden_bytes;
      const bool compact_records = region.delta_theta_final_capacity == 13 &&
          region.delta_theta_component_scratch_capacity == 13 &&
          final_bytes == 13 * sizeof(double) && component_bytes == 13 * sizeof(double) &&
          final_range_bytes == final_bytes && summary_range_bytes == 13 * sizeof(double) &&
          component_range_bytes == component_bytes;
      const bool permitted_global_k = reduction == expected_reductions &&
          global_k_bytes == expected_global_k && global_k_range_bytes == expected_global_k &&
          global_k_bytes < forbidden_bytes && global_k == (worker_id == 0 ? 1 : 0) &&
          global_k_derivative_bytes < forbidden_bytes;
      std::cout << "STREAMED_CALL_PATH_COUNTERS worker=" << worker_id
                << " b=" << b << " a=" << a << " m=" << m
                << " b_pair=" << pair << " reduction=" << reduction
                << " global_k=" << global_k << '\n';
      std::cout << "STREAMED_CALL_PATH_ALLOCATION_AUDIT worker=" << worker_id
                << " n=" << region.n << " m=" << region.m
                << " forbidden_25nn_bytes=" << forbidden_bytes
                << " b_a_m_bundle_bytes=0"
                << " pair_shape=3x(" << region.n << 'x' << region.m << ")+3x("
                << region.m << 'x' << region.m << ") pair_bytes=" << pair_bytes
                << " final_shape=13 final_bytes=" << final_bytes
                << " component_shape=13 component_bytes=" << component_bytes
                << " global_k_shape=" << region.delta_theta_reduction_scratch_capacity
                << " global_k_bytes=" << global_k_bytes
                << " global_k_derivative_shape=5x(" << region.m << 'x' << region.m
                << ") global_k_derivative_bytes=" << global_k_derivative_bytes
                << " cuda_range_bytes(final,summary,component,global_k)="
                << final_range_bytes << ',' << summary_range_bytes << ','
                << component_range_bytes << ',' << global_k_range_bytes << '\n';
      valid = valid && no_legacy_bundle && compact_pair && compact_records &&
          permitted_global_k;
    }
    std::cout << "STREAMED_CALL_PATH_LEGACY_B_A_M_ALLOCATIONS=" << (valid ? "ZERO" : "FAIL")
              << " STREAMED_CALL_PATH_COMPLETE_PRODUCER=" << (valid ? "PASS" : "FAIL")
              << " STREAMED_CALL_PATH_NO_25NN_ARENA=" << (valid ? "PASS" : "FAIL") << '\n';
    return valid ? 0 : 1;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
