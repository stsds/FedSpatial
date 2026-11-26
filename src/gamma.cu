#include "fedspatial/gpu_context.hpp"
#include <sstream>
#include <stdexcept>

namespace fedspatial {
namespace {
void check_gamma(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
const char* cublas_status_name(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    default: return "CUBLAS_STATUS_UNKNOWN";
  }
}
void check_gamma_blas(GpuContext& context, cublasStatus_t status, const char* operation,
                      int m, int n, int k, int lda, int ldb, int ldc,
                      size_t a_capacity, size_t b_capacity, size_t c_capacity) {
  std::ostringstream detail;
  detail << "cuBLAS gamma " << operation << " status=" << cublas_status_name(status)
         << "(" << int(status) << ") device=" << context.device_id()
         << " m=" << m << " n=" << n << " k=" << k
         << " lda=" << lda << " ldb=" << ldb << " ldc=" << ldc
         << " capacities={A:" << a_capacity << ",B:" << b_capacity
         << ",C:" << c_capacity << "} doubles";
  if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(detail.str());
  const cudaError_t sync = cudaStreamSynchronize(context.stream());
  if (sync != cudaSuccess) throw std::runtime_error(detail.str() + " async=" + cudaGetErrorString(sync));
  const cudaError_t async = cudaGetLastError();
  if (async != cudaSuccess) throw std::runtime_error(detail.str() + " last_error=" + cudaGetErrorString(async));
}
__global__ void gamma_rhs(const double* z, const double* b, const double* mu, double* rhs, int n, int m) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double value = z[i];
  // Device B is the row-major n-by-m storage also consumed as B' by cuBLAS.
  for (int column = 0; column < m; ++column) value -= b[i * m + column] * mu[column];
  rhs[i] = value;
}
}

DeviceGammaLocalSummary compute_device_gamma_local(GpuContext& context, const WorkerInput& worker,
    const Matrix& knots, const GlobalParameters& parameters) {
  PProfileScope profile_scope(context.profile(), "compute_device_gamma_local");
  const DeviceMuSigmaLocalState local = build_device_mu_sigma_local(context, worker, knots, parameters);
  const int n = local.n, q = local.q;
  if (worker.z.cols != 1 || parameters.gamma.rows != q || parameters.gamma.cols != 1)
    throw std::runtime_error("gamma input shape");
  check_gamma(cudaSetDevice(context.device_id()));
  auto& region = const_cast<RegionGpuState&>(context.region());
  const double one = 1., zero = 0.;
  gamma_rhs<<<(n + 63) / 64, 64, 0, context.stream()>>>(region.z, local.B,
      local.mu, region.rhs, n, local.m);
  check_gamma(cudaGetLastError());
  check_gamma(cudaMemcpyAsync(region.gamma_x_solved, region.x_col_major, size_t(n) * q * sizeof(double),
                               cudaMemcpyDeviceToDevice, context.stream()));
  check_gamma_blas(context, cublasDtrsm(context.cublas(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
      CUBLAS_DIAG_NON_UNIT, n, q, &one, local.chol_M, n, region.gamma_x_solved, n), "DTRSM(N)", n, q, 0, n, n, n, size_t(n)*n, size_t(n)*q, size_t(n)*q);
  check_gamma_blas(context, cublasDtrsm(context.cublas(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
      CUBLAS_DIAG_NON_UNIT, n, q, &one, local.chol_M, n, region.gamma_x_solved, n), "DTRSM(T)", n, q, 0, n, n, n, size_t(n)*n, size_t(n)*q, size_t(n)*q);
  check_gamma_blas(context, cublasDtrsm(context.cublas(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
      CUBLAS_DIAG_NON_UNIT, n, 1, &one, local.chol_M, n, region.rhs, n), "DTRSM-vector(N)", n, 1, 0, n, n, n, size_t(n)*n, size_t(n), size_t(n));
  check_gamma_blas(context, cublasDtrsm(context.cublas(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
      CUBLAS_DIAG_NON_UNIT, n, 1, &one, local.chol_M, n, region.rhs, n), "DTRSM-vector(T)", n, 1, 0, n, n, n, size_t(n)*n, size_t(n), size_t(n));
  check_gamma_blas(context, cublasDgemm(context.cublas(), CUBLAS_OP_T, CUBLAS_OP_N, q, 1, n, &one,
      region.x_col_major, n, region.rhs, n, &zero, region.gamma_local, q), "DGEMM-XT-rhs", q, 1, n, n, n, q, size_t(n)*q, size_t(n), size_t(q));
  check_gamma_blas(context, cublasDgemm(context.cublas(), CUBLAS_OP_T, CUBLAS_OP_N, q, q, n, &one,
      region.x_col_major, n, region.gamma_x_solved, n, &zero, region.gamma_xx, q), "DGEMM-XT-X", q, q, n, n, n, q, size_t(n)*q, size_t(n)*q, size_t(q)*q);
  return {q, region.gamma_local, region.gamma_xx};
}

GammaLocalSummary compute_gamma_local(GpuContext& context, const WorkerInput& worker,
    const Matrix& knots, const GlobalParameters& parameters) {
  const auto device = compute_device_gamma_local(context, worker, knots, parameters);
  GammaLocalSummary output{Matrix(device.q, 1), Matrix(device.q, device.q)};
  check_gamma(cudaMemcpyAsync(output.gamma.a.data(), device.gamma, size_t(device.q) * sizeof(double),
      cudaMemcpyDeviceToHost, context.stream()));
  check_gamma(cudaMemcpyAsync(output.xx.a.data(), device.xx, size_t(device.q) * device.q * sizeof(double),
      cudaMemcpyDeviceToHost, context.stream()));
  check_gamma(cudaStreamSynchronize(context.stream()));
  context.record_model_d2h_bytes((size_t(device.q) + size_t(device.q) * device.q) * sizeof(double));
  return output;
}
}  // namespace fedspatial
