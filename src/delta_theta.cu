#include "fedspatial/gpu_context.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace fedspatial {
namespace {
std::mutex producer_cache_log_mutex;
thread_local DeviceDeltaThetaProducerBoundaryAudit* producer_boundary_audit=nullptr;
void capture_producer_boundary(GpuContext& context,int boundary) {
  if(!producer_boundary_audit) return;
  const auto& r=context.region();
  if(cudaMemcpyAsync(producer_boundary_audit->summary[boundary].data(),r.delta_theta_summary,
      11*sizeof(double),cudaMemcpyDeviceToHost,context.stream())!=cudaSuccess ||
     cudaMemcpyAsync(producer_boundary_audit->final[boundary].data(),r.delta_theta_final,
      13*sizeof(double),cudaMemcpyDeviceToHost,context.stream())!=cudaSuccess ||
     cudaStreamSynchronize(context.stream())!=cudaSuccess) throw std::runtime_error("producer boundary audit copy");
}
int producer_tile_width(int m) {
  const char* disabled=std::getenv("FEDSPATIAL_DISABLE_TILED_SOLVES");
  if (disabled && disabled[0] && disabled[0]!='0') return 1;
  // All panel consumers and their red-zoned owned buffers are dynamically
  // sized up to 1024 columns.  The historical fixed 32 left most available
  // capacity idle at m=400 and doubled the identical panel-launch sequence.
  // Performance mode uses the largest validated panel and caps it by m.
  // The environment override retains smaller deterministic diagnostic and
  // memory-bounded configurations.
  int requested=512;
  if(const char* value=std::getenv("FEDSPATIAL_PRODUCER_TILE_WIDTH"); value&&value[0]) {
    requested=std::atoi(value);
    if(requested<1||requested>1024) throw std::runtime_error("FEDSPATIAL_PRODUCER_TILE_WIDTH must be in [1,1024]");
  }
  return std::min(requested,m);
}
bool mixed_trace_tile_enabled() {
  const char* enabled=std::getenv("FEDSPATIAL_MIXED_TRACE_TILED_ACTION");
  return enabled && enabled[0] && enabled[0]!='0';
}
bool mixed_contractions_tile_enabled() {
  const char* enabled=std::getenv("FEDSPATIAL_MIXED_CONTRACTIONS_TILED");
  return enabled && enabled[0] && enabled[0]!='0';
}
bool theta_first_tile_enabled() {
  const char* enabled=std::getenv("FEDSPATIAL_THETA_FIRST_TILED");
  return enabled && enabled[0] && enabled[0]!='0';
}
bool theta_hessian_aa_log_tile_enabled() {
  const char* enabled=std::getenv("FEDSPATIAL_THETA_HESSIAN_AA_LOG_TILED");
  return enabled && enabled[0] && enabled[0]!='0';
}
bool theta_hessian_ar_log_tile_enabled() { const char* e=std::getenv("FEDSPATIAL_THETA_HESSIAN_AR_LOG_TILED");return e&&e[0]&&e[0]!='0'; }
bool theta_hessian_rr_log_tile_enabled() { const char* e=std::getenv("FEDSPATIAL_THETA_HESSIAN_RR_LOG_TILED");return e&&e[0]&&e[0]!='0'; }
bool theta_hessian_q_w_tile_enabled() { const char* e=std::getenv("FEDSPATIAL_THETA_HESSIAN_Q_W_TILED");return e&&e[0]&&e[0]!='0'; }
bool theta_hessian_q_panel_tile_enabled() { const char* e=std::getenv("FEDSPATIAL_THETA_HESSIAN_Q_PANEL_TILED");return e&&e[0]&&e[0]!='0'; }
// Direct family/Python gates passed with device-resident pointer arrays.
// Set FEDSPATIAL_THETA_HESSIAN_Q_SUPERPANEL=0 for the Q-base-only fallback.
bool theta_hessian_q_superpanel_enabled() { const char* e=std::getenv("FEDSPATIAL_THETA_HESSIAN_Q_SUPERPANEL");return !(e&&e[0]&&e[0]=='0'); }
// The direct scalar/tiled/Python gates establish this as the default path.
// FEDSPATIAL_THETA_HESSIAN_Q_BASE_REUSE=0 restores the canonical repeated
// action sequence for focused diagnostics.
bool theta_hessian_q_base_reuse_enabled() { const char* e=std::getenv("FEDSPATIAL_THETA_HESSIAN_Q_BASE_REUSE");return !(e&&e[0]&&e[0]=='0'); }
// The repaired matrix-RHS second-action path is the production default.
// Set FEDSPATIAL_THETA_HESSIAN_SECOND_LOG_TILED=0 for canonical scalar
// actions when diagnosing parity on a new platform.
bool theta_hessian_second_log_tile_enabled() { const char* e=std::getenv("FEDSPATIAL_THETA_HESSIAN_SECOND_LOG_TILED");return !(e&&e[0]&&e[0]=='0'); }
// Validated production default.  Set =0 to retain the canonical scalar
// triple-product launch sequence for a direct reference/oracle run.
bool theta_second_triple_batched_enabled() { const char* e=std::getenv("FEDSPATIAL_THETA_SECOND_TRIPLE_BATCHED");return !(e&&e[0]&&e[0]=='0'); }
bool theta_hessian_second_log_fused_reduction_enabled() { const char* e=std::getenv("FEDSPATIAL_THETA_HESSIAN_SECOND_LOG_FUSED_REDUCTION");return e&&e[0]&&e[0]!='0'; }
// Validated performance defaults.  Setting either variable to 0 restores the
// direct per-RHS covariance action as a focused FP64 oracle/debug path.
bool theta_cov_first_matrix_enabled() { const char* e=std::getenv("FEDSPATIAL_THETA_COV_FIRST_MATRIX");return !(e&&e[0]&&e[0]=='0'); }
bool theta_cov_second_matrix_enabled() { const char* e=std::getenv("FEDSPATIAL_THETA_COV_SECOND_MATRIX");return !(e&&e[0]&&e[0]=='0'); }
class ProducerStateCacheScope {
 public:
  explicit ProducerStateCacheScope(GpuContext& context) : context_(context) {
    auto& r=const_cast<RegionGpuState&>(context_.region());
    r.producer_state_cache_active=true;
    r.producer_state_cache_valid=false;
    builds_=r.producer_state_build_count;
    hits_=r.producer_state_cache_hit_count;
    b_reuses_=r.producer_b_base_reuse_count;
    scalar_solves_=r.producer_scalar_solve_calls;
    tiled_solves_=r.producer_tiled_solve_calls;
    tiled_columns_=r.producer_tiled_rhs_columns;
    covariance_tiles_=r.producer_covariance_action_tiles;
    action_gemms_=r.producer_theta_action_gemm_calls;
    second_actions_=r.producer_second_derivative_actions;
    action_calls_=r.mixed_action_tile_calls;
    action_rhs_=r.mixed_action_tile_rhs_columns;
    trace_tiles_=r.mixed_action_trace_tile_calls;
    theta_first_tiles_=r.theta_first_trace_tile_calls;
    q_tiles_=r.theta_hessian_q_tile_calls;
    q_panel_tiles_=r.theta_hessian_q_panel_tile_calls;
    q_panel_rhs_=r.theta_hessian_q_panel_rhs_columns;
    q_panel_gemms_=r.theta_hessian_q_panel_gemm_calls;
    second_tile_calls_=r.theta_second_tile_calls;
    second_tile_columns_=r.theta_second_tile_rhs_columns;
    second_tile_allocations_=r.theta_second_tile_allocation_count;
    second_tile_gemms_=r.theta_second_tile_gemm_calls;
    second_batched_logical_gemms_=r.theta_second_batched_logical_gemm_calls;
    second_batched_actual_gemms_=r.theta_second_batched_actual_gemm_calls;
    second_batched_allocations_=r.theta_second_batched_allocation_count;
    second_batched_guards_=r.theta_second_batched_guard_checks;
    q_base_logical_actions_=r.theta_q_base_logical_actions;
    q_base_actual_actions_=r.theta_q_base_actual_actions;
    q_base_logical_solves_=r.theta_q_base_logical_solves;
    q_base_actual_solves_=r.theta_q_base_actual_solves;
    q_base_copies_=r.theta_q_base_cache_copies;
    q_base_allocations_=r.theta_q_base_cache_allocation_count;
    q_base_guards_=r.theta_q_base_cache_guard_checks;
    q_super_logical_gemms_=r.theta_q_super_logical_gemm_calls;
    q_super_actual_gemms_=r.theta_q_super_actual_gemm_calls;
    q_super_copies_=r.theta_q_super_output_copies;
    q_super_allocations_=r.theta_q_super_allocation_count;
    q_super_guards_=r.theta_q_super_guard_checks;
    dense_w_builds_=r.local_dense_w_builds;
    dense_w_reuses_=r.local_dense_w_reuses;
    dense_w_gemms_=r.local_dense_w_gemm_calls;
    dense_w_avoided_trsm_=r.local_dense_w_avoided_trsm;
    dense_w_fallbacks_=r.local_dense_w_fallbacks;
  }
  ~ProducerStateCacheScope() {
    auto& r=const_cast<RegionGpuState&>(context_.region());
    std::lock_guard<std::mutex> lock(producer_cache_log_mutex);
    std::cerr << "DELTA_THETA_STATE_CACHE device=" << context_.device_id()
              << " builds=" << (r.producer_state_build_count-builds_)
              << " state_hits=" << (r.producer_state_cache_hit_count-hits_)
              << " b_base_reuses=" << (r.producer_b_base_reuse_count-b_reuses_)
              << " scalar_solve_calls=" << (r.producer_scalar_solve_calls-scalar_solves_)
              << " tiled_solve_calls=" << (r.producer_tiled_solve_calls-tiled_solves_)
              << " tiled_rhs_columns=" << (r.producer_tiled_rhs_columns-tiled_columns_)
              << " covariance_action_tiles=" << (r.producer_covariance_action_tiles-covariance_tiles_)
              << " theta_action_gemms=" << (r.producer_theta_action_gemm_calls-action_gemms_)
              << " second_derivative_actions=" << (r.producer_second_derivative_actions-second_actions_)
              << " action_tiles=" << (r.mixed_action_tile_calls-action_calls_)
              << " action_rhs_columns=" << (r.mixed_action_tile_rhs_columns-action_rhs_)
              << " mixed_trace_tiles=" << (r.mixed_action_trace_tile_calls-trace_tiles_)
              << " theta_first_trace_tiles=" << (r.theta_first_trace_tile_calls-theta_first_tiles_)
              << " theta_hessian_q_tiles=" << (r.theta_hessian_q_tile_calls-q_tiles_)
              << " theta_hessian_q_panel_tiles=" << (r.theta_hessian_q_panel_tile_calls-q_panel_tiles_)
              << " theta_hessian_q_panel_rhs_columns=" << (r.theta_hessian_q_panel_rhs_columns-q_panel_rhs_)
              << " theta_hessian_q_panel_gemms=" << (r.theta_hessian_q_panel_gemm_calls-q_panel_gemms_)
              << " second_action_tiles=" << (r.theta_second_tile_calls-second_tile_calls_)
              << " second_action_rhs_columns=" << (r.theta_second_tile_rhs_columns-second_tile_columns_)
              << " second_action_gemms=" << (r.theta_second_tile_gemm_calls-second_tile_gemms_)
              << " second_action_tile_allocations=" << (r.theta_second_tile_allocation_count-second_tile_allocations_)
              << " second_action_tile_n_capacity=" << r.theta_second_tile_n_capacity
              << " second_action_tile_m_capacity=" << r.theta_second_tile_m_capacity
              << " second_batched_logical_gemms=" << (r.theta_second_batched_logical_gemm_calls-second_batched_logical_gemms_)
              << " second_batched_actual_gemms=" << (r.theta_second_batched_actual_gemm_calls-second_batched_actual_gemms_)
              << " second_batched_allocations=" << (r.theta_second_batched_allocation_count-second_batched_allocations_)
              << " second_batched_guard_checks=" << (r.theta_second_batched_guard_checks-second_batched_guards_)
              << " second_batched_m_capacity=" << r.theta_second_batched_m_capacity
              << " q_base_logical_actions=" << (r.theta_q_base_logical_actions-q_base_logical_actions_)
              << " q_base_actual_actions=" << (r.theta_q_base_actual_actions-q_base_actual_actions_)
              << " q_base_logical_solves=" << (r.theta_q_base_logical_solves-q_base_logical_solves_)
              << " q_base_actual_solves=" << (r.theta_q_base_actual_solves-q_base_actual_solves_)
              << " q_base_cache_copies=" << (r.theta_q_base_cache_copies-q_base_copies_)
              << " q_base_cache_allocations=" << (r.theta_q_base_cache_allocation_count-q_base_allocations_)
              << " q_base_cache_guard_checks=" << (r.theta_q_base_cache_guard_checks-q_base_guards_)
              << " q_base_cache_capacity=" << r.theta_q_base_cache_capacity
              << " q_super_logical_gemms=" << (r.theta_q_super_logical_gemm_calls-q_super_logical_gemms_)
              << " q_super_actual_gemms=" << (r.theta_q_super_actual_gemm_calls-q_super_actual_gemms_)
              << " q_super_output_copies=" << (r.theta_q_super_output_copies-q_super_copies_)
              << " q_super_allocations=" << (r.theta_q_super_allocation_count-q_super_allocations_)
              << " q_super_guard_checks=" << (r.theta_q_super_guard_checks-q_super_guards_)
              << " q_super_out_capacity=" << r.theta_q_super_out_capacity
              << " q_super_m_capacity=" << r.theta_q_super_m_capacity
              << " dense_w_builds=" << (r.local_dense_w_builds-dense_w_builds_)
              << " dense_w_reuses=" << (r.local_dense_w_reuses-dense_w_reuses_)
              << " dense_w_gemm_calls=" << (r.local_dense_w_gemm_calls-dense_w_gemms_)
              << " dense_w_avoided_trsm=" << (r.local_dense_w_avoided_trsm-dense_w_avoided_trsm_)
              << " dense_w_fallbacks=" << (r.local_dense_w_fallbacks-dense_w_fallbacks_)
              << std::endl;
    r.producer_state_cache_valid=false;
    r.producer_state_cache_active=false;
  }
 private:
  GpuContext& context_;
  long long builds_{}, hits_{}, b_reuses_{}, scalar_solves_{}, tiled_solves_{}, tiled_columns_{},
      covariance_tiles_{}, action_gemms_{}, second_actions_{};
  long long action_calls_{}, action_rhs_{}, trace_tiles_{}, theta_first_tiles_{}, q_tiles_{},
      q_panel_tiles_{}, q_panel_rhs_{}, q_panel_gemms_{}, second_tile_calls_{},
      second_tile_columns_{}, second_tile_allocations_{}, second_tile_gemms_{},
      second_batched_logical_gemms_{}, second_batched_actual_gemms_{},
      second_batched_allocations_{}, second_batched_guards_{},
      q_base_logical_actions_{}, q_base_actual_actions_{}, q_base_logical_solves_{},
      q_base_actual_solves_{}, q_base_copies_{}, q_base_allocations_{}, q_base_guards_{},
      q_super_logical_gemms_{}, q_super_actual_gemms_{}, q_super_copies_{},
      q_super_allocations_{}, q_super_guards_{}, dense_w_builds_{},
      dense_w_reuses_{}, dense_w_gemms_{}, dense_w_avoided_trsm_{},
      dense_w_fallbacks_{};
};

void axpy(Matrix& a, const Matrix& b, double scale = 1.0) {
  for (int i = 0; i < a.rows * a.cols; ++i) a.a[i] += scale * b.a[i];
}

Matrix scaled(Matrix a, double scale) {
  for (double& value : a.a) value *= scale;
  return a;
}

double trace(const Matrix& a) {
  double value = 0.0;
  for (int i = 0; i < a.rows; ++i) value += a(i, i);
  return value;
}

double inner(const Matrix& a, const Matrix& b) {
  double value = 0.0;
  for (int i = 0; i < a.rows * a.cols; ++i) value += a.a[i] * b.a[i];
  return value;
}

Matrix right_solve_spd(const Matrix& a, const Matrix& k) {
  return transpose(solve_spd(k, transpose(a)));
}

Matrix zero(int rows, int cols) { return Matrix(rows, cols); }

Matrix covariance_first(GpuContext& context, const Matrix& x, const Matrix& y,
                        const GlobalParameters& parameters, int parameter) {
  const auto derivatives = matern_with_derivatives(
      context, x, y, parameters.nu2, parameters.alpha, parameters.range);
  return parameter == 1 ? derivatives.alpha : derivatives.range;
}

Matrix q_first(const Matrix& b, const Matrix& bi, const Matrix& sigma) {
  Matrix result = matmul(matmul(bi, sigma), transpose(b));
  axpy(result, matmul(matmul(b, sigma), transpose(bi)));
  return result;
}

Matrix q_second(const Matrix& b, const Matrix& bi, const Matrix& bj,
                const Matrix& bij, const Matrix& sigma) {
  Matrix result = matmul(matmul(bij, sigma), transpose(b));
  axpy(result, matmul(matmul(b, sigma), transpose(bij)));
  axpy(result, matmul(matmul(bi, sigma), transpose(bj)));
  axpy(result, matmul(matmul(bj, sigma), transpose(bi)));
  return result;
}

double trace_product(const Matrix& a, const Matrix& b) {
  return trace(matmul(a, b));
}

}  // namespace

// Fixed-state P worker objective used by Python's delta/theta block:
// -n log(delta) + log|M| + delta * (r' M^-1 r + tr(M^-1 Q)).
// mu, Sigma, and gamma are deliberately held fixed in this routine.
DeltaThetaLocalResult compute_delta_theta_local_legacy_oracle(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters,
    const PGlobalDerivativeWorkspace& global_workspace) {
  PProfileScope profile_scope(context.profile(), "compute_delta_theta_local");
  const auto local = compute_mu_sigma_local(context, worker, knots, parameters);
  const int n = worker.sites.rows;
  const int m = knots.rows;
  const Matrix identity_n = identity(n);
  const Matrix covariance_k =
      matern_with_derivatives(context, knots, knots, parameters.nu2,
                              parameters.alpha, parameters.range)
          .value;

  Matrix residual_covariance = local.M;
  axpy(residual_covariance, identity_n, -1.0);
  residual_covariance = scaled(residual_covariance, 1.0 / parameters.delta);

  // r has Python's sign: X gamma - z + B mu.  The sign is immaterial to
  // the quadratic value but is material when differentiating r.
  Matrix r = matmul(local.B, parameters.mu);
  axpy(r, local.e, -1.0);
  // M has already been factored by the local GPU path.  Reuse that FP64
  // factor with two device TRSMs instead of rebuilding a host Cholesky.
  const Matrix w = gpu_right_spd_solve(context, identity_n, local.chol_M);
  const Matrix u = solve_spd(local.M, r);
  const Matrix q = matmul(matmul(local.B, parameters.sigma), transpose(local.B));
  const double quadratic = inner(r, u);
  const double trace_q = trace_product(w, q);

  // Reuse the independently parity-tested M and B second-derivative APIs.
  PDerivativeWorkspace derivatives;
  const auto cnk_workspace = make_cnk_derivative_workspace(context, worker, knots, parameters);
  derivatives.b_second = compute_b_second_derivatives(
      context, worker, knots, parameters, global_workspace, cnk_workspace);
  derivatives.m_first = compute_m_first_derivatives(context, worker, knots, parameters);
  derivatives.a_second = compute_a_second_derivatives_from_b(
      context, worker, knots, parameters, derivatives.b_second);
  derivatives.m_second = assemble_m_second_derivatives(
      context, derivatives.m_first, derivatives.a_second, parameters.delta);
  const auto& b_second = derivatives.b_second;
  const auto& m_first = derivatives.m_first;
  const auto& m_second = derivatives.m_second;

  std::array<Matrix, 3> b_first{zero(n, m), zero(n, m), zero(n, m)};
  for (int parameter = 1; parameter < 3; ++parameter) {
    const Matrix c_i = covariance_first(context, worker.sites, knots,
                                        parameters, parameter);
    const Matrix k_i = covariance_first(context, knots, knots, parameters,
                                        parameter);
    b_first[parameter] = right_solve_spd(c_i, covariance_k);
    axpy(b_first[parameter],
         right_solve_spd(matmul(local.B, k_i), covariance_k), -1.0);
  }

  std::array<Matrix, 3> r_first{zero(n, 1), zero(n, 1), zero(n, 1)};
  std::array<Matrix, 3> q_first_values{zero(n, n), zero(n, n), zero(n, n)};
  for (int parameter = 1; parameter < 3; ++parameter) {
    r_first[parameter] = matmul(b_first[parameter], parameters.mu);
    q_first_values[parameter] =
        q_first(local.B, b_first[parameter], parameters.sigma);
  }

  std::array<Matrix, 3> m1{m_first.delta, m_first.alpha, m_first.range};
  std::array<std::array<Matrix, 3>, 3> m2{{
      {m_second.dd, m_second.da, m_second.dr},
      {m_second.ad, m_second.aa, m_second.ar},
      {m_second.rd, m_second.ra, m_second.rr},
  }};

  std::array<std::array<Matrix, 3>, 3> b2{{
      {zero(n, m), zero(n, m), zero(n, m)},
      {zero(n, m), b_second.baa, b_second.bar},
      {zero(n, m), b_second.bra, b_second.brr},
  }};

  DeltaThetaLocalResult out;
  out.objective = -n * std::log(parameters.delta) +
                  std::log(parameters.delta) * n + local.logdet_R +
                  parameters.delta * (quadratic + trace_q);
  // local.logdet_R = log|M| - n log(delta), so the expression above is log|M|.

  std::array<double, 3> h{};
  std::array<double, 3> h_first{};
  for (int i = 0; i < 3; ++i) {
    h[i] = quadratic + trace_q;
    const double q_i = 2.0 * inner(r_first[i], u) -
                       inner(u, matmul(m1[i], u));
    const double t_i = trace_product(w, q_first_values[i]) -
                       trace(matmul(matmul(matmul(w, m1[i]), w), q));
    h_first[i] = q_i + t_i;
    out.gradient[i] = trace_product(w, m1[i]) +
                      (i == 0 ? -static_cast<double>(n) / parameters.delta : 0.0) +
                      (i == 0 ? h[i] : 0.0) + parameters.delta * h_first[i];
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      Matrix r_ij = matmul(b2[i][j], parameters.mu);
      Matrix q_ij = q_second(local.B, b_first[i], b_first[j], b2[i][j],
                             parameters.sigma);

      const Matrix w_rj = matmul(w, r_first[j]);
      const Matrix w_mj_u = matmul(matmul(w, m1[j]), u);
      const Matrix w_mi_u = matmul(matmul(w, m1[i]), u);
      const double quadratic_ij =
          2.0 * inner(r_ij, u) + 2.0 * inner(r_first[i], w_rj) -
          2.0 * inner(r_first[i], w_mj_u) -
          2.0 * inner(r_first[j], w_mi_u) +
          inner(u, matmul(matmul(m1[j], w), matmul(m1[i], u))) +
          inner(u, matmul(matmul(m1[i], w), matmul(m1[j], u))) -
          inner(u, matmul(m2[i][j], u));

      const double trace_ij =
          trace_product(w, q_ij) -
          trace(matmul(matmul(matmul(w, m1[j]), w), q_first_values[i])) -
          trace(matmul(matmul(matmul(w, m1[i]), w), q_first_values[j])) +
          trace(matmul(matmul(matmul(matmul(w, m1[j]), w), m1[i]),
                       matmul(w, q))) +
          trace(matmul(matmul(matmul(matmul(w, m1[i]), w), m1[j]),
                       matmul(w, q))) -
          trace(matmul(matmul(matmul(w, m2[i][j]), w), q));

      const double value =
          (i == 0 && j == 0
               ? static_cast<double>(n) / (parameters.delta * parameters.delta)
               : 0.0) +
          trace_product(w, m2[i][j]) -
          trace(matmul(matmul(matmul(w, m1[j]), w), m1[i])) +
          (j == 0 ? h_first[i] : 0.0) +
          (i == 0 ? h_first[j] : 0.0) +
          parameters.delta * (quadratic_ij + trace_ij);
      out.hessian[3 * i + j] = value;
      out.hessian[3 * j + i] = value;
    }
  }
  return out;
}

namespace {
// Public component APIs retain their small diagnostic downloads.  The
// synchronous producer flips this guard while composing its transport record:
// every component then remains entirely device-resident.
thread_local bool device_only_components = false;
struct DeviceOnlyComponentsScope {
  bool prior{device_only_components};
  DeviceOnlyComponentsScope() { device_only_components = true; }
  ~DeviceOnlyComponentsScope() { device_only_components = prior; }
};
void check_dt(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
void check_dt(cublasStatus_t x) { if (x != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS delta/theta streamed contraction failure"); }

// One output element per thread, with a fixed k order.  This deliberately
// avoids atomics and library reduction-order variation in the final P block.
__global__ void dt_matmul(const double* a, const double* b, double* c, int n) {
  const int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n*n) return;
  const int i=t/n,j=t%n; double value=0.;
  for(int k=0;k<n;++k) value+=a[i*n+k]*b[k*n+j];
  c[t]=value;
}
__device__ double dtrace_product(const double* a,const double* b,int n) {
  double v=0.; for(int i=0;i<n;++i) for(int j=0;j<n;++j) v+=a[i*n+j]*b[j*n+i]; return v;
}
__device__ double ddot(const double* a,const double* b,int n) { double v=0.;for(int i=0;i<n;++i)v+=a[i]*b[i];return v; }
__device__ double dquad_w(const double* a,const double* w,const double* b,int n) {
  double v=0.;for(int i=0;i<n;++i){double x=0.;for(int j=0;j<n;++j)x+=w[i*n+j]*b[j];v+=a[i]*x;}return v;
}
__device__ double dquad_m(const double* a,const double* m,const double* b,int n) {
  double v=0.;for(int i=0;i<n;++i){double x=0.;for(int j=0;j<n;++j)x+=m[i*n+j]*b[j];v+=a[i]*x;}return v;
}
__device__ double dquad_mw(const double* u,const double* mj,const double* wmi,const double* v,int n) {
  double result=0.;for(int i=0;i<n;++i){double x=0.;for(int j=0;j<n;++j){double y=0.;for(int k=0;k<n;++k)y+=wmi[j*n+k]*v[k];x+=mj[i*n+j]*y;}result+=u[i]*x;}return result;
}
__global__ void dt_reduce(const double* chol,const double* r,const double* u,const double* w,const double* q,
    const double* m0,const double* m1,const double* m2,
    const double* m00,const double* m01,const double* m02,const double* m10,const double* m11,const double* m12,const double* m20,const double* m21,const double* m22,
    const double* r0,const double* r1,const double* r2,const double* q0,const double* q1,const double* q2,
    const double* r00,const double* r01,const double* r02,const double* r10,const double* r11,const double* r12,const double* r20,const double* r21,const double* r22,
    const double* q00,const double* q01,const double* q02,const double* q10,const double* q11,const double* q12,const double* q20,const double* q21,const double* q22,
    const double* scratch,double delta,double* out,int n) {
  if(blockIdx.x||threadIdx.x)return;
  const double* mi[3]={m0,m1,m2}; const double* mij[9]={m00,m01,m02,m10,m11,m12,m20,m21,m22};
  const double* ri[3]={r0,r1,r2}; const double* qi[3]={q0,q1,q2};
  const double* rij[9]={r00,r01,r02,r10,r11,r12,r20,r21,r22}; const double* qij[9]={q00,q01,q02,q10,q11,q12,q20,q21,q22};
  const size_t nn=size_t(n)*n; const double* wm[3]={scratch,scratch+nn,scratch+2*nn}; const double* wmw[3]={scratch+3*nn,scratch+4*nn,scratch+5*nn};
  double logdet=-n*log(delta);for(int i=0;i<n;++i)logdet+=2.*log(chol[i*n+i]);
  const double h=ddot(r,u,n)+dtrace_product(w,q,n); out[0]=logdet+delta*h;
  double hp[3];
  for(int i=0;i<3;++i){const double qi1=2.*ddot(ri[i],u,n)-dquad_m(u,mi[i],u,n); const double ti=dtrace_product(w,qi[i],n)-dtrace_product(wmw[i],q,n); hp[i]=qi1+ti; out[1+i]=dtrace_product(w,mi[i],n)+(i==0?-double(n)/delta:0.)+(i==0?h:0.)+delta*hp[i];}
  for(int i=0;i<3;++i)for(int j=i;j<3;++j){const int ij=3*i+j;const double qijv=2.*ddot(rij[ij],u,n)+2.*dquad_w(ri[i],w,ri[j],n)-2.*dquad_w(ri[i],wm[j],u,n)-2.*dquad_w(ri[j],wm[i],u,n)+dquad_mw(u,mi[j],wm[i],u,n)+dquad_mw(u,mi[i],wm[j],u,n)-dquad_m(u,mij[ij],u,n);
    const double traceij=dtrace_product(w,qij[ij],n)-dtrace_product(wmw[j],qi[i],n)-dtrace_product(wmw[i],qi[j],n)+dtrace_product(scratch+(6+3*j+i)*nn,q,n)+dtrace_product(scratch+(6+3*i+j)*nn,q,n)-dtrace_product(scratch+(16+ij)*nn,q,n);
    const double value=(i==0&&j==0?double(n)/(delta*delta):0.)+dtrace_product(w,mij[ij],n)-dtrace_product(wmw[j],mi[i],n)+(j==0?hp[i]:0.)+(i==0?hp[j]:0.)+delta*(qijv+traceij);
    out[4+ij]=value;out[4+3*j+i]=value;
  }
}

// One block owns one tile and visits elements in a prescribed order.  The
// second kernel visits tile partials in launch-index order, so no atomics or
// library reduction tree can perturb a repeated FP64 result.
template<int Tile>
__global__ void streamed_trace_tiles(const double* left, const double* right,
                                     double* partials, int n) {
  const int tiles=(n+Tile-1)/Tile, tile=blockIdx.x;
  if (tile>=tiles*tiles || threadIdx.x) return;
  const int tr=tile/tiles, tc=tile%tiles;
  const int ib=tr*Tile, ie=min(n,ib+Tile), jb=tc*Tile, je=min(n,jb+Tile);
  double value=0.;
  for(int i=ib;i<ie;++i) for(int j=jb;j<je;++j) value+=left[i*n+j]*right[j*n+i];
  partials[tile]=value;
}
__global__ void streamed_sum(const double* partials, int count, double* out) {
  if(blockIdx.x||threadIdx.x)return; double value=0.;
  for(int i=0;i<count;++i)value+=partials[i]; *out=value;
}
__global__ void streamed_base_finish(const double* chol,const double* r,const double* u,
                                     const double* trace,double delta,int n,double* out) {
  if(blockIdx.x||threadIdx.x)return;
  double q=0.,logdet=-double(n)*log(delta);
  for(int i=0;i<n;++i) { q+=r[i]*u[i]; logdet+=2.*log(chol[i*n+i]); }
  out[0]=q; out[1]=*trace; out[2]=logdet+delta*(q+*trace);
}
// V is delta*W*B (the persistent minv_B state).  gram is V'V.  One thread
// fixes both summation orders; this is small (m*m), not an n*n product.
__global__ void streamed_delta_first_finish(const double* w,const double* u,
    const double* gram,const double* sigma,double delta,int n,int m,double* out) {
  if(blockIdx.x||threadIdx.x)return;
  double trw=0., uu=0., vv=0.;
  for(int i=0;i<n;++i) { trw+=w[i*n+i]; uu+=u[i]*u[i]; }
  for(int i=0;i<m;++i) for(int j=0;j<m;++j) vv+=gram[i*m+j]*sigma[j*m+i];
  // d/d delta {log|M|-n log(delta)+delta h}
  // = -tr(W)/delta + u'u + tr((W B) Sigma (W B)').
  out[3]=-trw/delta+uu+vv/(delta*delta);
}
__global__ void streamed_subtract(double* x,const double* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]-=y[i];}
__global__ void streamed_identity_tile(double* x,int n,int columns,int offset){int t=blockIdx.x*blockDim.x+threadIdx.x;if(t<n*columns){int i=t%n,j=t/n;x[t]=(i==offset+j)?1.:0.;}}
__global__ void streamed_trace_inverse_tile(const double* x,int n,int columns,int offset,double* out){if(blockIdx.x||threadIdx.x)return;double v=0.;for(int j=0;j<columns;++j)v+=x[(offset+j)+n*j];*out=v;}
__global__ void streamed_trace_bwb(const double* btb,const double* sigma,double delta,int m,double* out){if(blockIdx.x||threadIdx.x)return;double v=0.;for(int i=0;i<m;++i)for(int j=0;j<m;++j)v+=btb[i*m+j]*sigma[j*m+i];*out=v/delta;}
__global__ void streamed_delta_finish_scalar(double trw,const double* u,const double* gram,const double* sigma,double delta,int n,int m,double* out){if(blockIdx.x||threadIdx.x)return;double uu=0.,vv=0.;for(int i=0;i<n;++i)uu+=u[i]*u[i];for(int i=0;i<m;++i)for(int j=0;j<m;++j)vv+=gram[i*m+j]*sigma[j*m+i];out[3]=-trw/delta+uu+vv/(delta*delta);}
__global__ void streamed_delta_finish_scalar_device(const double* trw,const double* u,const double* gram,const double* sigma,double delta,int n,int m,double* out){if(blockIdx.x||threadIdx.x)return;double uu=0.,vv=0.;for(int i=0;i<n;++i)uu+=u[i]*u[i];for(int i=0;i<m;++i)for(int j=0;j<m;++j)vv+=gram[i*m+j]*sigma[j*m+i];out[3]=-*trw/delta+uu+vv/(delta*delta);}
__global__ void streamed_add_scalar(double* target,const double* value){if(!blockIdx.x&&!threadIdx.x)*target+=*value;}
__global__ void streamed_square_tile(const double* x,int n,int columns,double* out){if(blockIdx.x||threadIdx.x)return;double v=0.;for(int j=0;j<columns;++j)for(int i=0;i<n;++i){double z=x[i+n*j];v+=z*z;}*out=v;}
__global__ void streamed_dot_and_difference_norm(const double* u,const double* wu,int n,double* dot,double* norm){if(blockIdx.x||threadIdx.x)return;double d=0.,q=0.;for(int i=0;i<n;++i){d+=u[i]*wu[i];double z=u[i]-wu[i];q+=z*z;}*dot=d;*norm=q;}
__global__ void streamed_gram_trace_scaled(const double* gram,const double* sigma,int m,double scale,double* out){if(blockIdx.x||threadIdx.x)return;double v=0.;for(int i=0;i<m;++i)for(int j=0;j<m;++j)v+=gram[i*m+j]*sigma[j*m+i];*out=scale*v;}
__global__ void streamed_delta_hessian_finish(double trw,double trw2,double h,double hd_input,double u_dot_wu,double u_minus_wu_sq,double trw2q,double trw3q,double delta,int n,double* out){if(blockIdx.x||threadIdx.x)return;const double d2=delta*delta;const double uau=(hd_input-u_dot_wu)/delta;const double twawq=(h-trw2q)/delta;const double hd=-(uau+twawq);const double ldd=(double(n)- (double(n)-2.*trw+trw2))/d2;const double hdd=2.*(u_minus_wu_sq+(h-2.*trw2q+trw3q))/d2;out[4]=ldd+2.*hd+delta*hdd;}
__global__ void streamed_delta_hessian_from_components(const double* s,double delta,int n,double* out){if(blockIdx.x||threadIdx.x)return;const double trw=s[3],trw2=s[4],uwu=s[5],trw2q=s[7],trw3q=s[8];const double u2=out[1]+trw/delta-trw2q;const double uau=(s[0]-u2)/delta;const double twawq=(s[1]-trw2q)/delta;const double hd=-(uau+twawq);const double ldd=(2.*trw-trw2)/(delta*delta);const double hdd=2.*(s[0]-2.*u2+uwu+s[1]-2.*trw2q+trw3q)/(delta*delta);out[4]=ldd+2.*hd+delta*hdd;}

// The theta path intentionally evaluates A_i rather than constructing it.
// A=Cjj-B*Cnk' gives A_i=Cjj_i-B_i*Cnk'-B*Cnk_i'.  Keeping this identity in
// the element action is what removes the persistent A_i/M_i matrices.
__device__ double theta_cov_first(const double* x,int ix,const double* y,int iy,
                                  int nu2,double alpha,double range,int parameter) {
  const double dx=x[2*ix]-y[2*iy],dy=x[2*ix+1]-y[2*iy];
  double u=sqrt(dx*dx+dy*dy)/range,e,p,rb;
  if(nu2==1){e=exp(-u);p=e;rb=e*u/range;}
  else if(nu2==3){u*=sqrt(3.);e=exp(-u);p=(1.+u)*e;rb=e*u*u/range;}
  else {u*=sqrt(5.);e=exp(-u);p=(1.+u+u*u/3.)*e;rb=e*(u*u+u*u*u)/(3.*range);}
  return parameter==0?p:alpha*rb;
}
__device__ double theta_a_first(const double* sites,const double* knots,const double* b,
                                const double* bi,int row,int col,int n,int m,int nu2,
                                double alpha,double range,int parameter) {
  double value=theta_cov_first(sites,row,sites,col,nu2,alpha,range,parameter);
  for(int k=0;k<m;++k) {
    value-=bi[row*m+k]*theta_cov_first(sites,col,knots,k,nu2,alpha,range,2);
    value-=b[row*m+k]*theta_cov_first(sites,col,knots,k,nu2,alpha,range,parameter);
  }
  return value;
}
// parameter=2 selects the covariance value in the helper above.  This avoids
// a second value-only covariance kernel in the streamed action.
__device__ double theta_cov_value(const double* x,int ix,const double* y,int iy,
                                  int nu2,double alpha,double range) {
  const double dx=x[2*ix]-y[2*iy],dy=x[2*ix+1]-y[2*iy]; double u=sqrt(dx*dx+dy*dy)/range;
  if(nu2==1)return alpha*exp(-u);
  if(nu2==3){u*=sqrt(3.);return alpha*(1.+u)*exp(-u);}
  u*=sqrt(5.);return alpha*(1.+u+u*u/3.)*exp(-u);
}
// Correct the value branch used by theta_a_first without an extra matrix.
__device__ double theta_a_first_value(const double* sites,const double* knots,const double* b,
                                      const double* bi,int row,int col,int m,int nu2,
                                      double alpha,double range,int parameter) {
  double value=theta_cov_first(sites,row,sites,col,nu2,alpha,range,parameter);
  // Keep the K_i term explicit.  The shorter substitution through C_nk_i
  // is algebraically valid only when B_i and B are from the exact same
  // factorization/layout; the streamed first-B slice is independently
  // refreshed and that assumption caused the remaining theta mismatch.
  for(int a=0;a<m;++a) for(int q=0;q<m;++q) {
    const double kab=theta_cov_value(knots,a,knots,q,nu2,alpha,range);
    const double ki=theta_cov_first(knots,a,knots,q,nu2,alpha,range,parameter);
    value-=bi[row*m+a]*kab*b[col*m+q];
    value-=b[row*m+a]*ki*b[col*m+q];
    value-=b[row*m+a]*kab*bi[col*m+q];
  }
  return value;
}
// Equivalent differentiated Schur-complement form A=Cjj-B*Cnk'.  It avoids
// reconstructing K inside the action and, crucially, consumes the same
// freshly computed B_theta as the r/Q terms.
__device__ double theta_a_first_compact(const double* sites,const double* knots,const double* b,
                                        const double* bi,int row,int col,int m,int nu2,
                                        double alpha,double range,int parameter) {
  double value=theta_cov_first(sites,row,sites,col,nu2,alpha,range,parameter);
  for(int k=0;k<m;++k) {
    value-=bi[row*m+k]*theta_cov_value(sites,col,knots,k,nu2,alpha,range);
    value-=b[row*m+k]*theta_cov_first(sites,col,knots,k,nu2,alpha,range,parameter);
  }
  return value;
}
__global__ void theta_base_finish(const double* chol,const double* r,const double* u,const double* btb,
                                  const double* sigma,double delta,int n,int m,double* out) {
  if(blockIdx.x||threadIdx.x)return; double q=0.,t=0.,ld=-double(n)*log(delta);
  for(int i=0;i<n;++i){q+=r[i]*u[i];ld+=2.*log(chol[i*n+i]);}
  for(int i=0;i<m;++i)for(int j=0;j<m;++j)t+=btb[i*m+j]*sigma[j*m+i]/delta;
  out[0]=ld+delta*(q+t);
}
__global__ void theta_zero(double* x,int index){if(blockIdx.x||threadIdx.x)return;x[index]=0.;}
__global__ void theta_copy(const double* in,double* out){if(blockIdx.x||threadIdx.x)return;*out=*in;}
__global__ void theta_sum_copy(const double* in,int count,double* out){if(blockIdx.x||threadIdx.x)return;double v=0.;for(int i=0;i<count;++i)v+=in[i];*out=v;}
__global__ void theta_trace_column(const double* wcol,const double* mi,int col,int n,double* accum){if(blockIdx.x||threadIdx.x)return;double v=0.;for(int row=0;row<n;++row)v+=wcol[row]*mi[row*n+col];*accum+=v;}
__global__ void theta_quadratic(const double* u,const double* mu,const double* bi,const double* mi,int n,int m,double* ri_term,double* mi_term){if(blockIdx.x||threadIdx.x)return;double ri_u=0.,umu=0.;for(int i=0;i<n;++i){double ri=0.;for(int k=0;k<m;++k)ri+=bi[i*m+k]*mu[k];ri_u+=ri*u[i];double mu=0.;for(int j=0;j<n;++j)mu+=mi[i*n+j]*u[j];umu+=u[i]*mu;}*ri_term=2.*ri_u;*mi_term=umu;}
__global__ void theta_ri_u(const double* u,const double* mu,const double* bi,int n,int m,double* out){if(blockIdx.x||threadIdx.x)return;double value=0.;for(int i=0;i<n;++i){double ri=0.;for(int k=0;k<m;++k)ri+=bi[i*m+k]*mu[k];value+=2.*ri*u[i];}*out=value;}
// bi is the row-major n-by-m covariance derivative.  minvb, produced by
// cuBLAS/TRSM, is the column-major n-by-m V=delta*W*B.  Do not use the B
// stride for V: that silently aliases a different element when n != m.
__global__ void theta_t1(const double* bi,const double* minvb,const double* sigma,int n,int m,double delta,double* out){if(blockIdx.x||threadIdx.x)return;double v=0.;for(int a=0;a<m;++a)for(int b=0;b<m;++b){double z=0.;for(int i=0;i<n;++i)z+=bi[i*m+a]*minvb[i+n*b];v+=z*sigma[b*m+a];}*out=2.*v/delta;}
__global__ void theta_miv_column(const double* mi,const double* minvb,int source,double* out,int n){const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double v=0.;for(int j=0;j<n;++j)v+=mi[i*n+j]*minvb[j+n*source];out[i]=v;}
__global__ void theta_t2_column(const double* minvb,const double* y,const double* sigma,int source,int n,int m,double* accum){if(blockIdx.x||threadIdx.x)return;double v=0.;for(int a=0;a<m;++a){double dot=0.;for(int i=0;i<n;++i)dot+=minvb[i+n*a]*y[i];v+=dot*sigma[source*m+a];}*accum+=v;}
// Both tile reductions preserve the legacy source/column ascending order.
__global__ void theta_trace_tile_dot(const double* w,const double* mi,int n,int columns,int offset,double* accum){if(blockIdx.x||threadIdx.x)return;for(int local=0;local<columns;++local){double v=0.;for(int row=0;row<n;++row)v+=w[row+n*local]*mi[row+n*local];*accum+=v;}}
__global__ void theta_t2_tile_add(const double* gram,const double* sigma,int m,int offset,int columns,double* accum){if(blockIdx.x||threadIdx.x)return;for(int local=0;local<columns;++local){double v=0.;const int source=offset+local;for(int a=0;a<m;++a)v+=gram[a+m*local]*sigma[source*m+a];*accum+=v;}}
__global__ void theta_t1_gram(const double* gram,const double* sigma,int m,double delta,double* out){if(blockIdx.x||threadIdx.x)return;double v=0.;for(int a=0;a<m;++a)for(int b=0;b<m;++b)v+=gram[a+m*b]*sigma[b*m+a];*out=2.*v/delta;}
__global__ void theta_diag_tile_add(const double* x,int n,int columns,int offset,double scale,double* out){if(blockIdx.x||threadIdx.x)return;for(int local=0;local<columns;++local)*out+=scale*x[offset+local+local*n];}
// Exact launch-reduction helper for the scalar second action.  Each column is
// scaled elementwise before its diagonal is consumed; columns and the final
// accumulator are visited in the canonical increasing order.
__global__ void theta_scale_diag_tile_add(double* x,int n,int columns,int offset,double scale,double* out){if(blockIdx.x||threadIdx.x)return;for(int local=0;local<columns;++local){double* col=x+size_t(local)*n;for(int row=0;row<n;++row)col[row]*=scale;*out+=col[offset+local];}}
__global__ void theta_scale(double* x,double factor){if(blockIdx.x||threadIdx.x)return;*x*=factor;}
__global__ void theta_finish(const double* ri_term,const double* mi_term,const double* t1,const double* t2,double delta,int gradient_slot,double* out){if(blockIdx.x||threadIdx.x)return;out[gradient_slot]+=delta*(*ri_term-*mi_term+*t1-*t2);}

// One on-demand M_theta action.  Unlike compute_device_m_derivatives this
// never materializes A_theta/M_theta: b_first is n-by-m and the result is a
// single n-vector.  Keeping this action here also fixes its algebra to the
// independently validated streamed theta-first path.
__global__ void theta_mixed_action(const double* sites,const double* knots,const double* b,
    const double* bi,const double* x,double* out,int n,int m,int nu2,double delta,
    double alpha,double range,int parameter){const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;double v=0.;for(int col=0;col<n;++col)v+=delta*theta_a_first_value(sites,knots,b,bi,row,col,m,nu2,alpha,range,parameter)*x[col];out[row]=v;}
__global__ void mixed_trace_add(const double* z,int column,double delta,double* out){if(blockIdx.x||threadIdx.x)return;*out+=z[column]/delta;}
// One block deliberately preserves the old ascending source-column addition
// order while reading a compact n-by-tile solved action.
__global__ void mixed_trace_tile_add(const double* z,int n,int columns,int offset,double delta,double* out){if(blockIdx.x||threadIdx.x)return;for(int local=0;local<columns;++local)*out+=z[size_t(local)*n+offset+local]/delta;}
__global__ void mixed_vector_terms(const double* u,const double* wu,const double* wtu,
    const double* mu,const double* bi,int n,int m,double* rt_wu,double* u_wtu){if(blockIdx.x||threadIdx.x)return;double a=0.,b=0.;for(int i=0;i<n;++i){double ri=0.;for(int k=0;k<m;++k)ri+=bi[i*m+k]*mu[k];a+=ri*wu[i];b+=u[i]*wtu[i];}*rt_wu=2.*a;*u_wtu=-2.*b;}
__global__ void mixed_bi_w2b(const double* bi,const double* z,const double* sigma,int n,int m,double delta,double* out){if(blockIdx.x||threadIdx.x)return;double v=0.;for(int a=0;a<m;++a)for(int b=0;b<m;++b){double dot=0.;for(int i=0;i<n;++i)dot+=bi[i*m+a]*z[i+n*b];v+=dot*sigma[b*m+a];}*out=2.*v/delta;}
// `gram` is the GEMM result Bi' Z (m-by-m, column major).  Its serial
// summation deliberately retains the legacy a-then-b FP64 reduction order.
__global__ void mixed_bi_w2b_gram_add(const double* gram,const double* sigma,int m,double delta,double* out){if(blockIdx.x||threadIdx.x)return;double v=0.;for(int a=0;a<m;++a)for(int b=0;b<m;++b)v+=gram[a+m*b]*sigma[b*m+a];*out=2.*v/delta;}
__global__ void mixed_v_tz_add(const double* v,const double* tz,const double* sigma,int source,int n,int m,double delta,double* out){if(blockIdx.x||threadIdx.x)return;double value=0.;for(int a=0;a<m;++a){double dot=0.;for(int i=0;i<n;++i)dot+=v[i+n*a]*tz[i];value+=dot*sigma[source*m+a];}*out-=2.*value/(delta*delta);}
// Columns are source-ascending, with the inner coefficient order unchanged.
__global__ void mixed_v_tz_tile_add(const double* gram,const double* sigma,int m,int offset,int columns,double delta,double* out){if(blockIdx.x||threadIdx.x)return;for(int local=0;local<columns;++local){double value=0.;const int source=offset+local;for(int a=0;a<m;++a)value+=gram[a+m*local]*sigma[source*m+a];*out-=2.*value/(delta*delta);}}
__global__ void theta_cov_action(const double* sites,const double* x,int n,int nu2,double alpha,double range,int p,int q,int second,double* out){
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double s=0.;
  for(int j=0;j<n;++j){double dx=sites[2*i]-sites[2*j],dy=sites[2*i+1]-sites[2*j+1],u=sqrt(dx*dx+dy*dy)/range,e,da,dr,daa,dar,drr;
    if(nu2==1){e=exp(-u);da=e;dr=alpha*e*u/range;daa=0.;dar=e*u/range;drr=alpha*e*(u*u-2.*u)/(range*range);}
    else if(nu2==3){u*=sqrt(3.);e=exp(-u);da=(1.+u)*e;dr=alpha*e*u*u/range;daa=0.;dar=e*u*u/range;drr=alpha*e*(u*u*u-3.*u*u)/(range*range);}
    else {u*=sqrt(5.);e=exp(-u);da=(1.+u+u*u/3.)*e;dr=alpha*e*(u*u+u*u*u)/(3.*range);daa=0.;dar=e*(u*u+u*u*u)/(3.*range);drr=alpha*e*(u*u*u*u-3.*u*u*u-3.*u*u)/(3.*range*range);}
    const double v=second?(p==0&&q==0?daa:(p==1&&q==1?drr:dar)):(p==0?da:dr);s+=v*x[j];}out[i]=s;
}
__global__ void theta_cov_action_tile(const double* sites,const double* x,int n,int columns,int nu2,double alpha,double range,int p,double* out){
  const int t=blockIdx.x*blockDim.x+threadIdx.x,row=t%n,col=t/n;if(row>=n||col>=columns)return;double s=0.;
  for(int j=0;j<n;++j){double dx=sites[2*row]-sites[2*j],dy=sites[2*row+1]-sites[2*j+1],u=sqrt(dx*dx+dy*dy)/range,e,v;
    if(nu2==1){e=exp(-u);v=p==0?e:alpha*e*u/range;}else if(nu2==3){u*=sqrt(3.);e=exp(-u);v=p==0?(1.+u)*e:alpha*e*u*u/range;}else{u*=sqrt(5.);e=exp(-u);v=p==0?(1.+u+u*u/3.)*e:alpha*e*(u*u+u*u*u)/(3.*range);}s+=v*x[j+n*col];}out[row+n*col]=s;
}
__global__ void theta_cov_first_matrix_build(const double* sites,int n,int nu2,double alpha,double range,int p,double* matrix){
  const size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const size_t nn=size_t(n)*n;if(t>=nn)return;
  const int row=int(t/n),col=int(t-size_t(row)*n);
  const double dx=sites[2*row]-sites[2*col],dy=sites[2*row+1]-sites[2*col+1];
  double u=sqrt(dx*dx+dy*dy)/range,e,v;
  if(nu2==1){e=exp(-u);v=p==0?e:alpha*e*u/range;}
  else if(nu2==3){u*=sqrt(3.);e=exp(-u);v=p==0?(1.+u)*e:alpha*e*u*u/range;}
  else {u*=sqrt(5.);e=exp(-u);v=p==0?(1.+u+u*u/3.)*e:alpha*e*(u*u+u*u*u)/(3.*range);}
  matrix[t]=v;
}
// The second theta covariance action is mathematically the same row-by-row
// sum as theta_cov_action, but evaluates a bounded RHS tile.  It is used only
// by the gated log-determinant Hessian term; no n-by-n derivative covariance
// is retained.
__global__ void theta_cov_second_action_tile(const double* sites,const double* x,int n,int columns,int nu2,double alpha,double range,int p,int q,double* out){
  const int t=blockIdx.x*blockDim.x+threadIdx.x,row=t%n,col=t/n;if(row>=n||col>=columns)return;double s=0.;
  for(int j=0;j<n;++j){double dx=sites[2*row]-sites[2*j],dy=sites[2*row+1]-sites[2*j+1],u=sqrt(dx*dx+dy*dy)/range,e,v;
    if(nu2==1){e=exp(-u);v=(p==0&&q==0)?0.:((p==1&&q==1)?alpha*e*(u*u-2.*u)/(range*range):e*u/range);}
    else if(nu2==3){u*=sqrt(3.);e=exp(-u);v=(p==0&&q==0)?0.:((p==1&&q==1)?alpha*e*(u*u*u-3.*u*u)/(range*range):e*u*u/range);}
    else {u*=sqrt(5.);e=exp(-u);v=(p==0&&q==0)?0.:((p==1&&q==1)?alpha*e*(u*u*u*u-3.*u*u*u-3.*u*u)/(3.*range*range):e*(u*u+u*u*u)/(3.*range));}
    s+=v*x[j+n*col];}out[row+n*col]=s;
}
__global__ void theta_cov_second_matrix_build(const double* sites,int n,int nu2,double alpha,double range,int p,int q,double* matrix){
  const size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const size_t nn=size_t(n)*n;if(t>=nn)return;
  const int row=int(t/n),col=int(t-size_t(row)*n);
  const double dx=sites[2*row]-sites[2*col],dy=sites[2*row+1]-sites[2*col+1];
  double u=sqrt(dx*dx+dy*dy)/range,e,v;
  if(nu2==1){e=exp(-u);v=(p==0&&q==0)?0.:((p==1&&q==1)?alpha*e*(u*u-2.*u)/(range*range):e*u/range);}
  else if(nu2==3){u*=sqrt(3.);e=exp(-u);v=(p==0&&q==0)?0.:((p==1&&q==1)?alpha*e*(u*u*u-3.*u*u)/(range*range):e*u*u/range);}
  else {u*=sqrt(5.);e=exp(-u);v=(p==0&&q==0)?0.:((p==1&&q==1)?alpha*e*(u*u*u*u-3.*u*u*u-3.*u*u)/(3.*range*range):e*(u*u+u*u*u)/(3.*range));}
  matrix[t]=v;
}
__global__ void theta_zero_vector(double*x,int n){const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=0.;}
__global__ void theta_dot_add(const double*x,const double*y,int n,double scale,double*out){if(blockIdx.x||threadIdx.x)return;double v=0.;for(int i=0;i<n;++i)v+=x[i]*y[i];*out+=scale*v;}
__global__ void theta_column_add(const double*x,int col,double scale,double*out){if(blockIdx.x||threadIdx.x)return;*out+=scale*x[col];}
// Populate cuBLAS batched pointer tables entirely on the owning stream.
// This avoids any pageable-host lifetime ambiguity between consecutive tiles.
__global__ void theta_second_batched_pointer_setup(
    const double* b,const double* k,const double* second,const double* fi,const double* fj,
    const double* ki,const double* kj,const double* kij,const double* input,
    double* m0,double* m1,size_t stride,int stage,const double** a,const double** bb,double** cc) {
  const int t=threadIdx.x;if(t>=9)return;
  const double* right[9]={b,b,fj,b,b,fj,fi,fi,second};
  const double* middle[9]={k,kj,k,ki,kij,ki,k,kj,k};
  if(stage==0){a[t]=right[t];bb[t]=input;cc[t]=m0+size_t(t)*stride;}
  else {a[t]=middle[t];bb[t]=m0+size_t(t)*stride;cc[t]=m1+size_t(t)*stride;}
}

// Row-major L/R are consumed through the matching column-major transpose
// view.  This is L * middle * R' * x.  The two m-vectors borrow the already
// resident m-by-m local summary, not the legacy B derivative scratch.
void theta_triple_action(GpuContext& c,const double* left,const double* middle,const double* right,
                         const double* x,double* out,double scale) {
  const auto&r=c.region(); const double one=1.,zero=0.;
  double* v0=r.bt_rinv_b; double* v1=r.bt_rinv_b+r.m;
  check_dt(cublasDgemv(c.cublas(),CUBLAS_OP_N,r.m,r.n,&one,right,r.m,x,1,&zero,v0,1));
  check_dt(cublasDgemv(c.cublas(),CUBLAS_OP_N,r.m,r.m,&one,middle,r.m,v0,1,&zero,v1,1));
  // The first scratch has served as the m-vector input to `middle`; borrow
  // it now as an n-vector output.  This works even for m==1 and avoids an
  // out==temporary alias when an action's destination is r.bk.
  double* tmp=r.minv_e;
  check_dt(cublasDgemv(c.cublas(),CUBLAS_OP_T,r.m,r.n,&one,left,r.m,v1,1,&zero,tmp,1));
  check_dt(cublasDaxpy(c.cublas(),r.n,&scale,tmp,1,out,1));
}
LocalSolveStrategy local_solve_strategy(const GpuContext& c,const DeviceMuSigmaLocalState&) {
  const char* v=std::getenv("FEDSPATIAL_LOCAL_SOLVE_STRATEGY");
  // Typed configuration is the normal policy boundary.  Keep the legacy
  // environment spelling as a diagnostic override while launchers migrate.
  // The performance-oriented production policy builds W once and reuses it
  // across the many theta gradient/Hessian panel applications.  The
  // streamed strategy remains an explicit lower-memory fallback.
  if(!v || !v[0]) {
    const auto configuration=c.local_solve_configuration();
    if(configuration.strategy!=LocalSolveStrategy::auto_select) return configuration.strategy;
    return LocalSolveStrategy::dense_inverse_gemm;
  }
  if(std::string(v)=="streamed_trsm") return LocalSolveStrategy::streamed_trsm;
  if(std::string(v)=="dense_inverse_gemm") return LocalSolveStrategy::dense_inverse_gemm;
  if(std::string(v)!="auto") throw std::runtime_error("local_solve_strategy must be auto|dense_inverse_gemm|streamed_trsm");
  return LocalSolveStrategy::dense_inverse_gemm;
}
void local_dense_w_ensure(GpuContext& c,const DeviceMuSigmaLocalState&s) {
  auto&r=const_cast<RegionGpuState&>(c.region()); const size_t elems=size_t(s.n)*size_t(s.n);
  if(s.n<=0 || elems/size_t(s.n)!=size_t(s.n)) throw std::runtime_error("dense W size overflow");
  if(r.local_dense_w_capacity<elems) {
    if(r.local_dense_w_raw) check_dt(cudaFree(r.local_dense_w_raw));
    check_dt(cudaMalloc(&r.local_dense_w_raw,(elems+2)*sizeof(double)));
    r.local_dense_w=r.local_dense_w_raw+1; r.local_dense_w_capacity=elems; ++r.local_dense_w_allocations;
  }
  if(r.local_dense_w_valid) { ++r.local_dense_w_reuses; return; }
  constexpr double lo=0x1.23456789abcp+17,hi=-0x1.987654321fedcp-19;
  check_dt(cudaMemcpyAsync(r.local_dense_w_raw,&lo,sizeof(double),cudaMemcpyHostToDevice,c.stream()));
  check_dt(cudaMemcpyAsync(r.local_dense_w_raw+elems+1,&hi,sizeof(double),cudaMemcpyHostToDevice,c.stream()));
  streamed_identity_tile<<<(s.n*s.n+255)/256,256,0,c.stream()>>>(r.local_dense_w,s.n,s.n,0);
  const double one=1.;
  check_dt(cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,s.n,s.n,&one,s.chol_M,s.n,r.local_dense_w,s.n));
  check_dt(cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,s.n,s.n,&one,s.chol_M,s.n,r.local_dense_w,s.n));
  ++r.local_dense_w_builds; r.local_dense_w_valid=true;
}
void local_dense_solve_ensure(GpuContext& c,size_t elems) { auto&r=const_cast<RegionGpuState&>(c.region());if(r.local_dense_solve_capacity<elems){if(r.local_dense_solve_raw)check_dt(cudaFree(r.local_dense_solve_raw));check_dt(cudaMalloc(&r.local_dense_solve_raw,(elems+2)*sizeof(double)));r.local_dense_solve=r.local_dense_solve_raw+1;r.local_dense_solve_capacity=elems;} }
}  // namespace
void build_device_local_dense_w_for_test(GpuContext& c,const DeviceMuSigmaLocalState& s) {
  check_dt(cudaSetDevice(c.device_id()));
  local_dense_w_ensure(c,s);
}
namespace {
void theta_solve(GpuContext& c,const DeviceMuSigmaLocalState&s,double*x){auto&r=const_cast<RegionGpuState&>(c.region());if(r.producer_state_cache_active)++r.producer_scalar_solve_calls;const double one=1.,zero=0.;if(local_solve_strategy(c,s)==LocalSolveStrategy::dense_inverse_gemm){local_dense_w_ensure(c,s);local_dense_solve_ensure(c,s.n);check_dt(cublasDgemv(c.cublas(),CUBLAS_OP_N,s.n,s.n,&one,r.local_dense_w,s.n,x,1,&zero,r.local_dense_solve,1));check_dt(cudaMemcpyAsync(x,r.local_dense_solve,size_t(s.n)*sizeof(double),cudaMemcpyDeviceToDevice,c.stream()));++r.local_dense_w_gemm_calls;r.local_dense_w_avoided_trsm+=2;return;}check_dt(cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,s.n,1,&one,s.chol_M,s.n,x,s.n));check_dt(cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,s.n,1,&one,s.chol_M,s.n,x,s.n));}
// cnk is scratch after the cached B has been constructed.  Columns are still
// consumed and scalar-accumulated in ascending order after this batched solve.
void theta_solve_tile(GpuContext& c,const DeviceMuSigmaLocalState&s,double*x,int columns){auto&r=const_cast<RegionGpuState&>(c.region());if(r.producer_state_cache_active){++r.producer_tiled_solve_calls;r.producer_tiled_rhs_columns+=columns;}const double one=1.,zero=0.;if(local_solve_strategy(c,s)==LocalSolveStrategy::dense_inverse_gemm){local_dense_w_ensure(c,s);const size_t elems=size_t(s.n)*columns;local_dense_solve_ensure(c,elems);if(!gpu_precision_gemm(c,CUBLAS_OP_N,CUBLAS_OP_N,s.n,columns,s.n,r.local_dense_w,s.n,x,s.n,r.local_dense_solve,s.n))check_dt(cublasDgemm(c.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,s.n,columns,s.n,&one,r.local_dense_w,s.n,x,s.n,&zero,r.local_dense_solve,s.n));check_dt(cudaMemcpyAsync(x,r.local_dense_solve,elems*sizeof(double),cudaMemcpyDeviceToDevice,c.stream()));++r.local_dense_w_gemm_calls;r.local_dense_w_avoided_trsm+=2;return;}check_dt(cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,s.n,columns,&one,s.chol_M,s.n,x,s.n));check_dt(cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,s.n,columns,&one,s.chol_M,s.n,x,s.n));}
void theta_a_action(GpuContext& c,const DeviceMuSigmaLocalState&s,const double*bi,const double*ki,int p,
                    const double*x,double*out){const auto&r=c.region();theta_cov_action<<<(s.n+255)/256,256,0,c.stream()>>>(r.sites,x,s.n,r.parameter_nu2,r.parameter_alpha_value,r.parameter_range_value,p,p,0,out);theta_triple_action(c,bi,r.k,r.b,x,out,-1.);theta_triple_action(c,r.b,ki,r.b,x,out,-1.);theta_triple_action(c,r.b,r.k,bi,x,out,-1.);}
void theta_a_action_tile(GpuContext& c,const DeviceMuSigmaLocalState&s,const double*bi,
                         const double*ki,int p,const double*x,double*out,int columns) {
  auto&r=const_cast<RegionGpuState&>(c.region());
  const int tile=producer_tile_width(s.m);
  if(columns<1||columns>tile)throw std::runtime_error("invalid mixed action tile");
  ++r.mixed_action_tile_calls;r.mixed_action_tile_rhs_columns+=columns;
  const size_t ne=size_t(s.n)*tile,me=size_t(s.m)*tile;
  if(r.mixed_action_tile_n_capacity<ne) {
    if(r.mixed_action_tile_in)check_dt(cudaFree(r.mixed_action_tile_in));
    if(r.mixed_action_tile_out)check_dt(cudaFree(r.mixed_action_tile_out));
    check_dt(cudaMalloc(&r.mixed_action_tile_in,ne*sizeof(double)));
    check_dt(cudaMalloc(&r.mixed_action_tile_out,ne*sizeof(double)));
    r.mixed_action_tile_n_capacity=ne;++r.mixed_action_tile_allocation_count;
  }
  if(r.mixed_action_tile_m_capacity<me) {
    if(r.mixed_action_tile_m0)check_dt(cudaFree(r.mixed_action_tile_m0));
    if(r.mixed_action_tile_m1)check_dt(cudaFree(r.mixed_action_tile_m1));
    check_dt(cudaMalloc(&r.mixed_action_tile_m0,me*sizeof(double)));
    check_dt(cudaMalloc(&r.mixed_action_tile_m1,me*sizeof(double)));
    r.mixed_action_tile_m_capacity=me;++r.mixed_action_tile_allocation_count;
  }
  check_dt(cudaMemcpyAsync(r.mixed_action_tile_in,x,size_t(s.n)*columns*sizeof(double),
                           cudaMemcpyDeviceToDevice,c.stream()));
  const double one=1.,minus=-1.,zero=0.;
  if(theta_cov_first_matrix_enabled()) {
    const size_t nn=size_t(s.n)*s.n;
    if(r.theta_cov_first_matrix_capacity<nn) {
      for(int parameter=0;parameter<2;++parameter) {
        if(r.theta_cov_first_matrix[parameter])
          check_dt(cudaFree(r.theta_cov_first_matrix[parameter]));
        check_dt(cudaMalloc(&r.theta_cov_first_matrix[parameter],nn*sizeof(double)));
        r.theta_cov_first_matrix_valid[parameter]=false;
      }
      r.theta_cov_first_matrix_capacity=nn;
      r.theta_cov_first_matrix_allocations+=2;
    }
    const bool changed=r.theta_cov_first_matrix_nu2!=r.parameter_nu2||
      r.theta_cov_first_matrix_alpha!=r.parameter_alpha_value||
      r.theta_cov_first_matrix_range!=r.parameter_range_value||
      r.theta_cov_first_matrix_sites_fingerprint!=r.sites_fingerprint;
    if(changed)for(bool& valid:r.theta_cov_first_matrix_valid)valid=false;
    if(!r.theta_cov_first_matrix_valid[p]) {
      theta_cov_first_matrix_build<<<int((nn+255)/256),256,0,c.stream()>>>(
        r.sites,s.n,r.parameter_nu2,r.parameter_alpha_value,r.parameter_range_value,p,
        r.theta_cov_first_matrix[p]);
      r.theta_cov_first_matrix_valid[p]=true;
      r.theta_cov_first_matrix_nu2=r.parameter_nu2;
      r.theta_cov_first_matrix_alpha=r.parameter_alpha_value;
      r.theta_cov_first_matrix_range=r.parameter_range_value;
      r.theta_cov_first_matrix_sites_fingerprint=r.sites_fingerprint;
      ++r.theta_cov_first_matrix_builds;
    }
    if(c.precision_policy().mode!=ArithmeticPrecision::mixed_fp16_fp32||
       !gpu_precision_gemm(c,CUBLAS_OP_N,CUBLAS_OP_N,s.n,columns,s.n,
      r.theta_cov_first_matrix[p],s.n,r.mixed_action_tile_in,s.n,
      r.mixed_action_tile_out,s.n))
      check_dt(cublasDgemm(c.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,s.n,columns,s.n,&one,
        r.theta_cov_first_matrix[p],s.n,r.mixed_action_tile_in,s.n,&zero,
        r.mixed_action_tile_out,s.n));
    ++r.theta_cov_first_matrix_gemm_calls;
  } else {
    theta_cov_action_tile<<<(s.n*columns+255)/256,256,0,c.stream()>>>(
      r.sites,r.mixed_action_tile_in,s.n,columns,r.parameter_nu2,
      r.parameter_alpha_value,r.parameter_range_value,p,r.mixed_action_tile_out);
  }
  for(const auto term:std::array<std::array<const double*,3>,3>{{
      {{bi,r.k,r.b}},{{r.b,ki,r.b}},{{r.b,r.k,bi}}}}) {
    check_dt(cublasDgemm(c.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,s.m,columns,s.n,&one,
      term[2],s.m,r.mixed_action_tile_in,s.n,&zero,r.mixed_action_tile_m0,s.m));
    check_dt(cublasDgemm(c.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,s.m,columns,s.m,&one,
      term[1],s.m,r.mixed_action_tile_m0,s.m,&zero,r.mixed_action_tile_m1,s.m));
    check_dt(cublasDgemm(c.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,s.n,columns,s.m,&minus,
      term[0],s.m,r.mixed_action_tile_m1,s.m,&one,r.mixed_action_tile_out,s.n));
  }
  if(out!=r.mixed_action_tile_out)
    check_dt(cudaMemcpyAsync(out,r.mixed_action_tile_out,size_t(s.n)*columns*sizeof(double),
                             cudaMemcpyDeviceToDevice,c.stream()));
}
void theta_a_second_action(GpuContext& c,const DeviceMuSigmaLocalState&s,const DeviceBPairDerivativeSlice& d,int p,int q,const double*x,double*out){auto&r=const_cast<RegionGpuState&>(c.region());if(r.producer_state_cache_active)++r.producer_second_derivative_actions;const double*bi=d.first_i;const double*bj=d.first_j;const double*ki=d.k_i;const double*kj=d.k_j;const double*bij=d.second;const double*kij=d.k_ij;theta_cov_action<<<(s.n+255)/256,256,0,c.stream()>>>(r.sites,x,s.n,r.parameter_nu2,r.parameter_alpha_value,r.parameter_range_value,p,q,1,out);for(const auto&t:std::array<std::array<const double*,3>,9>{{{{bij,r.k,r.b}},{{bi,kj,r.b}},{{bi,r.k,bj}},{{bj,ki,r.b}},{{r.b,kij,r.b}},{{r.b,ki,bj}},{{bj,r.k,bi}},{{r.b,kj,bi}},{{r.b,r.k,bij}}}})theta_triple_action(c,t[0],t[1],t[2],x,out,-1.);}
constexpr double theta_second_guard_lo=0x1.23456789abcp+17;
constexpr double theta_second_guard_hi=-0x1.987654321fedcp-19;
bool theta_second_tile_guard_audit_enabled() {
  const char* e=std::getenv("FEDSPATIAL_THETA_SECOND_TILE_CHECK_GUARDS");
  return e&&e[0]&&e[0]!='0';
}
void theta_second_tile_set_guards(GpuContext& c,size_t ncap,size_t mcap) {
  auto&r=const_cast<RegionGpuState&>(c.region());
  for(auto pair:std::array<std::pair<double*,size_t>,4>{{
      {r.theta_second_tile_in_raw,ncap},{r.theta_second_tile_out_raw,ncap},
      {r.theta_second_tile_m0_raw,mcap},{r.theta_second_tile_m1_raw,mcap}}}) {
    check_dt(cudaMemcpyAsync(pair.first,&theta_second_guard_lo,sizeof(double),cudaMemcpyHostToDevice,c.stream()));
    check_dt(cudaMemcpyAsync(pair.first+pair.second+1,&theta_second_guard_hi,sizeof(double),cudaMemcpyHostToDevice,c.stream()));
  }
}
void theta_second_tile_check_guards(GpuContext& c,size_t ncap,size_t mcap) {
  auto&r=const_cast<RegionGpuState&>(c.region());
  std::array<double,8> guard{}; int at=0;
  for(auto pair:std::array<std::pair<double*,size_t>,4>{{
      {r.theta_second_tile_in_raw,ncap},{r.theta_second_tile_out_raw,ncap},
      {r.theta_second_tile_m0_raw,mcap},{r.theta_second_tile_m1_raw,mcap}}}) {
    check_dt(cudaMemcpyAsync(&guard[at++],pair.first,sizeof(double),cudaMemcpyDeviceToHost,c.stream()));
    check_dt(cudaMemcpyAsync(&guard[at++],pair.first+pair.second+1,sizeof(double),cudaMemcpyDeviceToHost,c.stream()));
  }
  check_dt(cudaStreamSynchronize(c.stream())); ++r.theta_second_tile_guard_checks;
  for(int i=0;i<8;i+=2) if(guard[i]!=theta_second_guard_lo||guard[i+1]!=theta_second_guard_hi)
    throw std::runtime_error("second theta action tile red-zone overwritten");
}
// The base-Q cache has its own red zones rather than borrowing the second
// action panel.  Q action uses the latter as transient GEMM scratch, so an
// alias here would silently corrupt a live branch input.
void theta_q_base_cache_ensure(GpuContext& c,size_t capacity) {
  auto&r=const_cast<RegionGpuState&>(c.region());
  if(r.theta_q_base_cache_capacity<capacity) {
    if(r.theta_q_base_cache_raw) check_dt(cudaFree(r.theta_q_base_cache_raw));
    check_dt(cudaMalloc(&r.theta_q_base_cache_raw,(capacity+2)*sizeof(double)));
    r.theta_q_base_cache=r.theta_q_base_cache_raw+1;
    r.theta_q_base_cache_capacity=capacity;
    ++r.theta_q_base_cache_allocation_count;
  }
  if(r.theta_q_base_cache==r.bk || r.theta_q_base_cache==r.cnk ||
     r.theta_q_base_cache==r.theta_second_tile_in || r.theta_q_base_cache==r.theta_second_tile_out)
    throw std::runtime_error("Q base cache aliases a live producer buffer");
  if(theta_second_tile_guard_audit_enabled()) {
    check_dt(cudaMemcpyAsync(r.theta_q_base_cache_raw,&theta_second_guard_lo,sizeof(double),cudaMemcpyHostToDevice,c.stream()));
    check_dt(cudaMemcpyAsync(r.theta_q_base_cache_raw+capacity+1,&theta_second_guard_hi,sizeof(double),cudaMemcpyHostToDevice,c.stream()));
  }
}
void theta_q_base_cache_check_guards(GpuContext& c) {
  auto&r=const_cast<RegionGpuState&>(c.region());
  if(!theta_second_tile_guard_audit_enabled()) return;
  std::array<double,2> guard{};
  check_dt(cudaMemcpyAsync(&guard[0],r.theta_q_base_cache_raw,sizeof(double),cudaMemcpyDeviceToHost,c.stream()));
  check_dt(cudaMemcpyAsync(&guard[1],r.theta_q_base_cache_raw+r.theta_q_base_cache_capacity+1,sizeof(double),cudaMemcpyDeviceToHost,c.stream()));
  check_dt(cudaStreamSynchronize(c.stream())); ++r.theta_q_base_cache_guard_checks;
  if(guard[0]!=theta_second_guard_lo || guard[1]!=theta_second_guard_hi)
    throw std::runtime_error("Q base cache red-zone overwritten");
}
void theta_q_super_ensure(GpuContext& c,size_t outcap,size_t mcap) {
  auto&r=const_cast<RegionGpuState&>(c.region());
  if(r.theta_q_super_out_capacity<outcap) {
    if(r.theta_q_super_out_raw) check_dt(cudaFree(r.theta_q_super_out_raw));
    check_dt(cudaMalloc(&r.theta_q_super_out_raw,(outcap+2)*sizeof(double)));
    r.theta_q_super_out=r.theta_q_super_out_raw+1; r.theta_q_super_out_capacity=outcap;
    ++r.theta_q_super_allocation_count;
  }
  if(r.theta_q_super_m_capacity<mcap) {
    if(r.theta_q_super_m0_raw) check_dt(cudaFree(r.theta_q_super_m0_raw));
    if(r.theta_q_super_m1_raw) check_dt(cudaFree(r.theta_q_super_m1_raw));
    check_dt(cudaMalloc(&r.theta_q_super_m0_raw,(mcap+2)*sizeof(double)));
    check_dt(cudaMalloc(&r.theta_q_super_m1_raw,(mcap+2)*sizeof(double)));
    r.theta_q_super_m0=r.theta_q_super_m0_raw+1; r.theta_q_super_m1=r.theta_q_super_m1_raw+1;
    r.theta_q_super_m_capacity=mcap; r.theta_q_super_allocation_count+=2;
  }
  if(!r.theta_q_super_a_array) {
    check_dt(cudaMalloc(&r.theta_q_super_a_array,8*sizeof(double*)));
    check_dt(cudaMalloc(&r.theta_q_super_b_array,8*sizeof(double*)));
    check_dt(cudaMalloc(&r.theta_q_super_c_array,8*sizeof(double*)));
    r.theta_q_super_allocation_count+=3;
  }
  for(double* p:{r.theta_q_super_out,r.theta_q_super_m0,r.theta_q_super_m1})
    if(p==r.bk||p==r.cnk||p==r.theta_q_base_cache||p==r.theta_second_tile_in||p==r.theta_second_tile_out)
      throw std::runtime_error("Q super-panel aliases a live producer buffer");
  if(theta_second_tile_guard_audit_enabled()) {
    for(auto pair:std::array<std::pair<double*,size_t>,3>{{{r.theta_q_super_out_raw,outcap},{r.theta_q_super_m0_raw,mcap},{r.theta_q_super_m1_raw,mcap}}}) {
      check_dt(cudaMemcpyAsync(pair.first,&theta_second_guard_lo,sizeof(double),cudaMemcpyHostToDevice,c.stream()));
      check_dt(cudaMemcpyAsync(pair.first+pair.second+1,&theta_second_guard_hi,sizeof(double),cudaMemcpyHostToDevice,c.stream()));
    }
  }
}
void theta_q_super_check_guards(GpuContext& c) {
  auto&r=const_cast<RegionGpuState&>(c.region()); if(!theta_second_tile_guard_audit_enabled())return;
  std::array<double,6> guard{}; int at=0;
  for(auto pair:std::array<std::pair<double*,size_t>,3>{{{r.theta_q_super_out_raw,r.theta_q_super_out_capacity},{r.theta_q_super_m0_raw,r.theta_q_super_m_capacity},{r.theta_q_super_m1_raw,r.theta_q_super_m_capacity}}}) {
    check_dt(cudaMemcpyAsync(&guard[at++],pair.first,sizeof(double),cudaMemcpyDeviceToHost,c.stream()));
    check_dt(cudaMemcpyAsync(&guard[at++],pair.first+pair.second+1,sizeof(double),cudaMemcpyDeviceToHost,c.stream()));
  }
  check_dt(cudaStreamSynchronize(c.stream())); ++r.theta_q_super_guard_checks;
  for(int k=0;k<6;k+=2)if(guard[k]!=theta_second_guard_lo||guard[k+1]!=theta_second_guard_hi)
    throw std::runtime_error("Q super-panel red-zone overwritten");
}
void theta_second_tile_ensure(GpuContext& c,size_t ncap,size_t mcap) {
  auto&r=const_cast<RegionGpuState&>(c.region());
  if(r.theta_second_tile_n_capacity<ncap) {
    if(r.theta_second_tile_in_raw) check_dt(cudaFree(r.theta_second_tile_in_raw));
    if(r.theta_second_tile_out_raw) check_dt(cudaFree(r.theta_second_tile_out_raw));
    check_dt(cudaMalloc(&r.theta_second_tile_in_raw,(ncap+2)*sizeof(double)));
    check_dt(cudaMalloc(&r.theta_second_tile_out_raw,(ncap+2)*sizeof(double)));
    r.theta_second_tile_in=r.theta_second_tile_in_raw+1;
    r.theta_second_tile_out=r.theta_second_tile_out_raw+1;
    r.theta_second_tile_n_capacity=ncap; ++r.theta_second_tile_allocation_count;
  }
  if(r.theta_second_tile_m_capacity<mcap) {
    if(r.theta_second_tile_m0_raw) check_dt(cudaFree(r.theta_second_tile_m0_raw));
    if(r.theta_second_tile_m1_raw) check_dt(cudaFree(r.theta_second_tile_m1_raw));
    check_dt(cudaMalloc(&r.theta_second_tile_m0_raw,(mcap+2)*sizeof(double)));
    check_dt(cudaMalloc(&r.theta_second_tile_m1_raw,(mcap+2)*sizeof(double)));
    r.theta_second_tile_m0=r.theta_second_tile_m0_raw+1;
    r.theta_second_tile_m1=r.theta_second_tile_m1_raw+1;
    r.theta_second_tile_m_capacity=mcap; ++r.theta_second_tile_allocation_count;
  }
  theta_second_tile_set_guards(c,r.theta_second_tile_n_capacity,r.theta_second_tile_m_capacity);
}
void theta_second_batched_ensure(GpuContext& c,size_t mcap) {
  auto&r=const_cast<RegionGpuState&>(c.region());
  if(r.theta_second_batched_m_capacity<mcap) {
    if(r.theta_second_batched_m0_raw)check_dt(cudaFree(r.theta_second_batched_m0_raw));
    if(r.theta_second_batched_m1_raw)check_dt(cudaFree(r.theta_second_batched_m1_raw));
    check_dt(cudaMalloc(&r.theta_second_batched_m0_raw,(mcap+2)*sizeof(double)));
    check_dt(cudaMalloc(&r.theta_second_batched_m1_raw,(mcap+2)*sizeof(double)));
    r.theta_second_batched_m0=r.theta_second_batched_m0_raw+1;
    r.theta_second_batched_m1=r.theta_second_batched_m1_raw+1;
    r.theta_second_batched_m_capacity=mcap; r.theta_second_batched_allocation_count+=2;
  }
  if(!r.theta_second_batched_a_array) {
    check_dt(cudaMalloc(&r.theta_second_batched_a_array,9*sizeof(double*)));
    check_dt(cudaMalloc(&r.theta_second_batched_b_array,9*sizeof(double*)));
    check_dt(cudaMalloc(&r.theta_second_batched_c_array,9*sizeof(double*)));
    r.theta_second_batched_allocation_count+=3;
  }
  for(double* ptr:{r.theta_second_batched_m0,r.theta_second_batched_m1})
    if(ptr==r.bk||ptr==r.cnk||ptr==r.theta_second_tile_in||ptr==r.theta_second_tile_out||ptr==r.theta_q_base_cache)
      throw std::runtime_error("batched second-action scratch aliases live storage");
  // Guard the allocation boundary, not this call's (possibly smaller) tile
  // request: a context retains the largest capacity across remainder tiles.
  if(theta_second_tile_guard_audit_enabled()) for(auto pair:std::array<std::pair<double*,size_t>,2>{{{r.theta_second_batched_m0_raw,r.theta_second_batched_m_capacity},{r.theta_second_batched_m1_raw,r.theta_second_batched_m_capacity}}}) {
    check_dt(cudaMemcpyAsync(pair.first,&theta_second_guard_lo,sizeof(double),cudaMemcpyHostToDevice,c.stream()));
    check_dt(cudaMemcpyAsync(pair.first+pair.second+1,&theta_second_guard_hi,sizeof(double),cudaMemcpyHostToDevice,c.stream()));
  }
}
void theta_second_batched_check_guards(GpuContext& c) {
  auto&r=const_cast<RegionGpuState&>(c.region());if(!theta_second_tile_guard_audit_enabled())return;
  std::array<double,4> guard{};int at=0;for(auto pair:std::array<std::pair<double*,size_t>,2>{{{r.theta_second_batched_m0_raw,r.theta_second_batched_m_capacity},{r.theta_second_batched_m1_raw,r.theta_second_batched_m_capacity}}}) {check_dt(cudaMemcpyAsync(&guard[at++],pair.first,sizeof(double),cudaMemcpyDeviceToHost,c.stream()));check_dt(cudaMemcpyAsync(&guard[at++],pair.first+pair.second+1,sizeof(double),cudaMemcpyDeviceToHost,c.stream()));}
  check_dt(cudaStreamSynchronize(c.stream()));++r.theta_second_batched_guard_checks;
  for(int k=0;k<4;k+=2)if(guard[k]!=theta_second_guard_lo||guard[k+1]!=theta_second_guard_hi) {
    std::cerr << "batched second-action red-zone overwritten buffer=" << (k/2)
              << " lo=" << std::hexfloat << guard[k] << " hi=" << guard[k+1]
              << " expected_lo=" << theta_second_guard_lo << " expected_hi=" << theta_second_guard_hi
              << std::defaultfloat << " capacity=" << r.theta_second_batched_m_capacity << '\n';
    throw std::runtime_error("batched second-action red-zone overwritten");
  }
}
// Candidate M_ij action.  Its four dedicated, red-zoned buffers deliberately
// do not overlap W/bk/cnk or the first-action tiles.  GEMM changes reduction
// association, so callers keep this opt-in until full-producer parity passes.
void theta_a_second_action_tile(GpuContext& c,const DeviceMuSigmaLocalState&s,const DeviceBPairDerivativeSlice&d,int p,int q,const double*x,double*out,int columns){
  auto&r=const_cast<RegionGpuState&>(c.region());
  if(columns<1||columns>1024)throw std::runtime_error("invalid second action tile width");
  const size_t ne=size_t(s.n)*columns,me=size_t(s.m)*columns;
  theta_second_tile_ensure(c,ne,me); ++r.theta_second_tile_calls; r.theta_second_tile_rhs_columns+=columns;
  check_dt(cudaMemcpyAsync(r.theta_second_tile_in,x,ne*sizeof(double),cudaMemcpyDeviceToDevice,c.stream()));
  if(theta_cov_second_matrix_enabled()) {
    const size_t nn=size_t(s.n)*s.n;
    if(r.theta_cov_second_matrix_capacity<nn) {
      for(int index=0;index<3;++index) {
        if(r.theta_cov_second_matrix[index])check_dt(cudaFree(r.theta_cov_second_matrix[index]));
        check_dt(cudaMalloc(&r.theta_cov_second_matrix[index],nn*sizeof(double)));
        r.theta_cov_second_matrix_valid[index]=false;
      }
      r.theta_cov_second_matrix_capacity=nn;
      r.theta_cov_second_matrix_allocations+=3;
    }
    const bool changed=r.theta_cov_second_matrix_nu2!=r.parameter_nu2||
      r.theta_cov_second_matrix_alpha!=r.parameter_alpha_value||
      r.theta_cov_second_matrix_range!=r.parameter_range_value||
      r.theta_cov_second_matrix_sites_fingerprint!=r.sites_fingerprint;
    if(changed)for(bool& valid:r.theta_cov_second_matrix_valid)valid=false;
    const int index=p==0?(q==0?0:1):2;
    if(!r.theta_cov_second_matrix_valid[index]) {
      theta_cov_second_matrix_build<<<int((nn+255)/256),256,0,c.stream()>>>(r.sites,s.n,
        r.parameter_nu2,r.parameter_alpha_value,r.parameter_range_value,p,q,
        r.theta_cov_second_matrix[index]);
      r.theta_cov_second_matrix_valid[index]=true;
      r.theta_cov_second_matrix_nu2=r.parameter_nu2;
      r.theta_cov_second_matrix_alpha=r.parameter_alpha_value;
      r.theta_cov_second_matrix_range=r.parameter_range_value;
      r.theta_cov_second_matrix_sites_fingerprint=r.sites_fingerprint;
      ++r.theta_cov_second_matrix_builds;
    }
    const double one=1.,zero=0.;
    if(c.precision_policy().mode!=ArithmeticPrecision::mixed_fp16_fp32||
       !gpu_precision_gemm(c,CUBLAS_OP_N,CUBLAS_OP_N,s.n,columns,s.n,
      r.theta_cov_second_matrix[index],s.n,r.theta_second_tile_in,s.n,
      r.theta_second_tile_out,s.n))
      check_dt(cublasDgemm(c.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,s.n,columns,s.n,&one,
        r.theta_cov_second_matrix[index],s.n,r.theta_second_tile_in,s.n,&zero,
        r.theta_second_tile_out,s.n));
    ++r.theta_cov_second_matrix_gemm_calls;
  } else theta_cov_second_action_tile<<<(s.n*columns+255)/256,256,0,c.stream()>>>(r.sites,r.theta_second_tile_in,s.n,columns,r.parameter_nu2,r.parameter_alpha_value,r.parameter_range_value,p,q,r.theta_second_tile_out);
  const double one=1.,minus=-1.,zero=0.;
  const std::array<std::array<const double*,3>,9> terms={{{d.second,r.k,r.b},{d.first_i,d.k_j,r.b},{d.first_i,r.k,d.first_j},{d.first_j,d.k_i,r.b},{r.b,d.k_ij,r.b},{r.b,d.k_i,d.first_j},{d.first_j,r.k,d.first_i},{r.b,d.k_j,d.first_i},{r.b,r.k,d.second}}};
  if(theta_second_triple_batched_enabled()) {
    // The direct term oracle deliberately supplies widths beyond the current
    // producer tile.  Each of the nine disjoint regions must therefore be
    // sized for the actual RHS panel, not merely the normal traversal width.
    const int tile=std::max(producer_tile_width(s.m),columns);const size_t stride=size_t(s.m)*tile;
    theta_second_batched_ensure(c,9*stride);
    std::array<const double*,9> left{};std::array<double*,9> m1{};
    for(int t=0;t<9;++t){left[t]=terms[t][0];m1[t]=r.theta_second_batched_m1+size_t(t)*stride;}
    theta_second_batched_pointer_setup<<<1,32,0,c.stream()>>>(r.b,r.k,d.second,d.first_i,d.first_j,d.k_i,d.k_j,d.k_ij,r.theta_second_tile_in,r.theta_second_batched_m0,r.theta_second_batched_m1,stride,0,r.theta_second_batched_a_array,r.theta_second_batched_b_array,r.theta_second_batched_c_array);
    check_dt(cudaGetLastError());
    check_dt(cublasDgemmBatched(c.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,s.m,columns,s.n,&one,r.theta_second_batched_a_array,s.m,r.theta_second_batched_b_array,s.n,&zero,r.theta_second_batched_c_array,s.m,9));
    theta_second_batched_pointer_setup<<<1,32,0,c.stream()>>>(r.b,r.k,d.second,d.first_i,d.first_j,d.k_i,d.k_j,d.k_ij,r.theta_second_tile_in,r.theta_second_batched_m0,r.theta_second_batched_m1,stride,1,r.theta_second_batched_a_array,r.theta_second_batched_b_array,r.theta_second_batched_c_array);
    check_dt(cudaGetLastError());
    check_dt(cublasDgemmBatched(c.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,s.m,columns,s.m,&one,r.theta_second_batched_a_array,s.m,r.theta_second_batched_b_array,s.m,&zero,r.theta_second_batched_c_array,s.m,9));
    r.theta_second_batched_logical_gemm_calls+=27;r.theta_second_batched_actual_gemm_calls+=2;r.theta_second_tile_gemm_calls+=2;
    for(int t=0;t<9;++t){check_dt(cublasDgemm(c.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,s.n,columns,s.m,&minus,left[t],s.m,m1[t],s.m,&one,r.theta_second_tile_out,s.n));++r.theta_second_batched_actual_gemm_calls;++r.theta_second_tile_gemm_calls;}
    theta_second_batched_check_guards(c);
  } else for(const auto& term:terms){
    check_dt(cublasDgemm(c.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,s.m,columns,s.n,&one,term[2],s.m,r.theta_second_tile_in,s.n,&zero,r.theta_second_tile_m0,s.m));
    check_dt(cublasDgemm(c.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,s.m,columns,s.m,&one,term[1],s.m,r.theta_second_tile_m0,s.m,&zero,r.theta_second_tile_m1,s.m));
    check_dt(cublasDgemm(c.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,s.n,columns,s.m,&minus,term[0],s.m,r.theta_second_tile_m1,s.m,&one,r.theta_second_tile_out,s.n));
    r.theta_second_tile_gemm_calls+=3;
  }
  if(out!=r.theta_second_tile_out)check_dt(cudaMemcpyAsync(out,r.theta_second_tile_out,ne*sizeof(double),cudaMemcpyDeviceToDevice,c.stream()));
  if(theta_second_tile_guard_audit_enabled()) theta_second_tile_check_guards(c,r.theta_second_tile_n_capacity,r.theta_second_tile_m_capacity);
}
void theta_q_action(GpuContext& c,const double*left,const double*right,const double*x,double*out){const auto&r=c.region();theta_zero_vector<<<(r.n+255)/256,256,0,c.stream()>>>(out,r.n);theta_triple_action(c,left,r.parameter_sigma,right,x,out,1.);}
void theta_q_second_action(GpuContext& c,const DeviceBPairDerivativeSlice&d,const double*x,double*out){const auto&r=c.region();theta_zero_vector<<<(r.n+255)/256,256,0,c.stream()>>>(out,r.n);theta_triple_action(c,d.second,r.parameter_sigma,r.b,x,out,1.);theta_triple_action(c,r.b,r.parameter_sigma,d.second,x,out,1.);theta_triple_action(c,d.first_i,r.parameter_sigma,d.first_j,x,out,1.);theta_triple_action(c,d.first_j,r.parameter_sigma,d.first_i,x,out,1.);}

// Independent Q/W actions have no covariance-kernel contribution.  Collect
// their RHS columns in the same dedicated red-zoned panel used by M_ij; this
// buffer is never bk/cnk or the first-action tile.  The caller still reduces
// diagonal entries in increasing global-column order.
void theta_q_triple_panel(GpuContext& c,const DeviceMuSigmaLocalState&s,
                          const std::array<std::array<const double*,3>,4>& terms,int count,
                          const double*x,double*out,int columns) {
  auto&r=const_cast<RegionGpuState&>(c.region());
  if(columns<1||columns>1024||count<1||count>4) throw std::runtime_error("invalid Q action tile");
  const size_t ne=size_t(s.n)*columns, me=size_t(s.m)*columns;
  theta_second_tile_ensure(c,ne,me); ++r.theta_hessian_q_panel_tile_calls;
  r.theta_hessian_q_panel_rhs_columns+=columns;
  check_dt(cudaMemcpyAsync(r.theta_second_tile_in,x,ne*sizeof(double),cudaMemcpyDeviceToDevice,c.stream()));
  check_dt(cudaMemsetAsync(r.theta_second_tile_out,0,ne*sizeof(double),c.stream()));
  const double one=1.,zero=0.;
  for(int t=0;t<count;++t) {
    const auto& term=terms[t];
    check_dt(cublasDgemm(c.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,s.m,columns,s.n,&one,term[2],s.m,r.theta_second_tile_in,s.n,&zero,r.theta_second_tile_m0,s.m));
    check_dt(cublasDgemm(c.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,s.m,columns,s.m,&one,term[1],s.m,r.theta_second_tile_m0,s.m,&zero,r.theta_second_tile_m1,s.m));
    check_dt(cublasDgemm(c.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,s.n,columns,s.m,&one,term[0],s.m,r.theta_second_tile_m1,s.m,&one,r.theta_second_tile_out,s.n));
    r.theta_hessian_q_panel_gemm_calls+=3;
  }
  if(out!=r.theta_second_tile_out) check_dt(cudaMemcpyAsync(out,r.theta_second_tile_out,ne*sizeof(double),cudaMemcpyDeviceToDevice,c.stream()));
  if(theta_second_tile_guard_audit_enabled()) theta_second_tile_check_guards(c,r.theta_second_tile_n_capacity,r.theta_second_tile_m_capacity);
}
void theta_q_action_tile(GpuContext& c,const DeviceMuSigmaLocalState&s,const double*left,const double*right,const double*x,double*out,int columns){
  const auto&r=c.region(); const std::array<std::array<const double*,3>,4> terms={{{left,r.parameter_sigma,right},{nullptr,nullptr,nullptr},{nullptr,nullptr,nullptr},{nullptr,nullptr,nullptr}}};
  theta_q_triple_panel(c,s,terms,1,x,out,columns);
}
void theta_q_first_action_tile(GpuContext& c,const DeviceMuSigmaLocalState&s,const double*first,const double*x,double*out,int columns){
  const auto&r=c.region(); const std::array<std::array<const double*,3>,4> terms={{{first,r.parameter_sigma,r.b},{r.b,r.parameter_sigma,first},{nullptr,nullptr,nullptr},{nullptr,nullptr,nullptr}}};
  theta_q_triple_panel(c,s,terms,2,x,out,columns);
}
void theta_q_second_action_tile(GpuContext& c,const DeviceMuSigmaLocalState&s,const DeviceBPairDerivativeSlice&d,const double*x,double*out,int columns){
  const auto&r=c.region(); const std::array<std::array<const double*,3>,4> terms={{{d.second,r.parameter_sigma,r.b},{r.b,r.parameter_sigma,d.second},{d.first_i,r.parameter_sigma,d.first_j},{d.first_j,r.parameter_sigma,d.first_i}}};
  theta_q_triple_panel(c,s,terms,4,x,out,columns);
}

// Eight independent triple products sharing the same W-panel.  The first two
// GEMM stages are batched only across disjoint m-by-tile regions.  Final GEMMs
// stay in the canonical family/term order and accumulate to distinct output
// regions, so neither a terminal trace reduction nor a family output is
// reassociated.  Super out blocks are [Q_ij, Q_i, Q_j].
void theta_q_second_first_superpanel(GpuContext& c,const DeviceMuSigmaLocalState&s,
                                    const DeviceBPairDerivativeSlice&d,const double*x,int columns) {
  auto&r=const_cast<RegionGpuState&>(c.region()); const int tile=producer_tile_width(s.m);
  const size_t out_stride=size_t(s.n)*tile, m_stride=size_t(s.m)*tile;
  theta_q_super_ensure(c,3*out_stride,8*m_stride);
  const std::array<std::array<const double*,3>,8> terms={{
    {d.second,r.parameter_sigma,r.b},{r.b,r.parameter_sigma,d.second},
    {d.first_i,r.parameter_sigma,d.first_j},{d.first_j,r.parameter_sigma,d.first_i},
    {d.first_i,r.parameter_sigma,r.b},{r.b,r.parameter_sigma,d.first_i},
    {d.first_j,r.parameter_sigma,r.b},{r.b,r.parameter_sigma,d.first_j}}};
  std::array<const double*,8> right{},middle{},left{},input{},m0_const{};
  std::array<double*,8> m0{},m1{};
  for(int t=0;t<8;++t){right[t]=terms[t][2];middle[t]=terms[t][1];left[t]=terms[t][0];input[t]=x;m0[t]=r.theta_q_super_m0+size_t(t)*m_stride;m0_const[t]=m0[t];m1[t]=r.theta_q_super_m1+size_t(t)*m_stride;}
  const double one=1.,zero=0.;
  // CUDA 12.4's batched interface dereferences its pointer arrays from the
  // device.  Copy the small pointer tables to owned device arrays before the
  // stream-ordered GEMMs; stack arrays are host staging only.
  check_dt(cudaMemcpyAsync(r.theta_q_super_a_array,right.data(),8*sizeof(double*),cudaMemcpyHostToDevice,c.stream()));
  check_dt(cudaMemcpyAsync(r.theta_q_super_b_array,input.data(),8*sizeof(double*),cudaMemcpyHostToDevice,c.stream()));
  check_dt(cudaMemcpyAsync(r.theta_q_super_c_array,m0.data(),8*sizeof(double*),cudaMemcpyHostToDevice,c.stream()));
  check_dt(cublasDgemmBatched(c.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,s.m,columns,s.n,&one,
                              r.theta_q_super_a_array,s.m,r.theta_q_super_b_array,s.n,&zero,r.theta_q_super_c_array,s.m,8));
  check_dt(cudaMemcpyAsync(r.theta_q_super_a_array,middle.data(),8*sizeof(double*),cudaMemcpyHostToDevice,c.stream()));
  check_dt(cudaMemcpyAsync(r.theta_q_super_b_array,m0_const.data(),8*sizeof(double*),cudaMemcpyHostToDevice,c.stream()));
  check_dt(cudaMemcpyAsync(r.theta_q_super_c_array,m1.data(),8*sizeof(double*),cudaMemcpyHostToDevice,c.stream()));
  check_dt(cublasDgemmBatched(c.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,s.m,columns,s.m,&one,
                              r.theta_q_super_a_array,s.m,r.theta_q_super_b_array,s.m,&zero,r.theta_q_super_c_array,s.m,8));
  r.theta_q_super_logical_gemm_calls+=24; r.theta_q_super_actual_gemm_calls+=2;
  for(int t=0;t<8;++t) {
    const int family=t<4?0:(t<6?1:2); const double beta=(t==0||t==4||t==6)?0.:1.;
    check_dt(cublasDgemm(c.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,s.n,columns,s.m,&one,
                         left[t],s.m,m1[t],s.m,&beta,r.theta_q_super_out+size_t(family)*out_stride,s.n));
    ++r.theta_q_super_actual_gemm_calls;
  }
}

void streamed_fixed_vectors(GpuContext& c,const DeviceMuSigmaLocalState& s){auto&r=const_cast<RegionGpuState&>(c.region());const double one=1.,zero=0.;if(!r.delta_theta_r){check_dt(cudaMalloc(&r.delta_theta_r,size_t(s.n)*sizeof(double)));check_dt(cudaMalloc(&r.delta_theta_u,size_t(s.n)*sizeof(double)));++r.delta_theta_workspace_allocation_count;}check_dt(cublasDgemv(c.cublas(),CUBLAS_OP_T,s.m,s.n,&one,s.B,s.m,s.mu,1,&zero,r.delta_theta_r,1));streamed_subtract<<<(s.n+255)/256,256,0,c.stream()>>>(r.delta_theta_r,s.e,s.n);check_dt(cudaMemcpyAsync(r.delta_theta_u,r.delta_theta_r,size_t(s.n)*sizeof(double),cudaMemcpyDeviceToDevice,c.stream()));check_dt(cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,s.n,1,&one,s.chol_M,s.n,r.delta_theta_u,s.n));check_dt(cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,s.n,1,&one,s.chol_M,s.n,r.delta_theta_u,s.n));}
double streamed_trace_inverse(GpuContext& c,const DeviceMuSigmaLocalState&s){const int tile=std::min(128,s.m);auto&r=const_cast<RegionGpuState&>(c.region());double one=1.,total=0.;theta_zero<<<1,1,0,c.stream()>>>(r.delta_theta_summary,3);for(int offset=0;offset<s.n;offset+=tile){int cols=std::min(tile,s.n-offset);streamed_identity_tile<<<(s.n*cols+255)/256,256,0,c.stream()>>>(r.bk,s.n,cols,offset);check_dt(cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,s.n,cols,&one,s.chol_M,s.n,r.bk,s.n));check_dt(cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,s.n,cols,&one,s.chol_M,s.n,r.bk,s.n));streamed_trace_inverse_tile<<<1,1,0,c.stream()>>>(r.bk,s.n,cols,offset,r.delta_theta_summary+10);streamed_add_scalar<<<1,1,0,c.stream()>>>(r.delta_theta_summary+3,r.delta_theta_summary+10);}if(device_only_components)return 0.;check_dt(cudaMemcpyAsync(&total,r.delta_theta_summary+3,sizeof(double),cudaMemcpyDeviceToHost,c.stream()));check_dt(cudaStreamSynchronize(c.stream()));return total;}
}

DeviceDeltaThetaBaseReduction compute_device_delta_theta_base_reduction_streamed(
    GpuContext& context,const WorkerInput& worker,const Matrix& knots,
    const GlobalParameters& parameters) {
  const auto state=build_device_mu_sigma_local(context,worker,knots,parameters);
  streamed_fixed_vectors(context,state);
  auto& region=context.region_;
  // Three scalar slots share the existing 13-double summary allocation.  No
  // n-by-n temporary is introduced by this contraction family.
  if(!region.delta_theta_summary) check_dt(cudaMalloc(&region.delta_theta_summary,13*sizeof(double)));
  streamed_trace_bwb<<<1,1,0,context.stream_>>>(state.bt_minv_B,state.sigma,
      parameters.delta,state.m,region.delta_theta_summary+1);
  streamed_base_finish<<<1,1,0,context.stream_>>>(state.chol_M,region.delta_theta_r,region.delta_theta_u,
      region.delta_theta_summary+1,parameters.delta,state.n,region.delta_theta_summary);
  check_dt(cudaGetLastError());
  if (device_only_components) return {};
  std::array<double,3> host{};
  check_dt(cudaMemcpyAsync(host.data(),region.delta_theta_summary,3*sizeof(double),cudaMemcpyDeviceToHost,context.stream_));
  check_dt(cudaStreamSynchronize(context.stream_)); region.result_d2h_bytes+=3*sizeof(double);
  return {host[0],host[1],host[2]};
}

DeviceDeltaThetaDeltaFirstReduction
compute_device_delta_theta_delta_first_reduction_streamed(
    GpuContext& context,const WorkerInput& worker,const Matrix& knots,
    const GlobalParameters& parameters) {
  const auto base=compute_device_delta_theta_base_reduction_streamed(
      context,worker,knots,parameters);
  // minv_B is delta*W*B from build_device_mu_sigma_local.  Reuse the compact
  // m*m local summary buffer for V'V after every consumer of B'W B is done.
  const auto state=build_device_mu_sigma_local(context,worker,knots,parameters);
  auto& region=context.region_;
  const double one=1.,zero=0.;
  check_dt(cublasDgemm(context.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,state.m,state.m,state.n,
                        &one,state.minv_B,state.n,state.minv_B,state.n,&zero,
                        region.bt_rinv_b,state.m));
  const double trw=streamed_trace_inverse(context,state);
  if (device_only_components)
    streamed_delta_finish_scalar_device<<<1,1,0,context.stream()>>>(region.delta_theta_summary+3,
        region.delta_theta_u,region.bt_rinv_b,state.sigma,parameters.delta,state.n,state.m,
        region.delta_theta_summary);
  else
    streamed_delta_finish_scalar<<<1,1,0,context.stream()>>>(trw,region.delta_theta_u,region.bt_rinv_b,
        state.sigma,parameters.delta,state.n,state.m,region.delta_theta_summary);
  check_dt(cudaGetLastError());
  if (device_only_components) return {};
  double gradient=0.;
  check_dt(cudaMemcpyAsync(&gradient,region.delta_theta_summary+3,sizeof(double),
                           cudaMemcpyDeviceToHost,context.stream()));
  check_dt(cudaStreamSynchronize(context.stream_));
  region.result_d2h_bytes+=sizeof(double);
  return {base.objective,gradient};
}

DeviceDeltaThetaDeltaHessianReduction
compute_device_delta_theta_delta_hessian_reduction_streamed(
    GpuContext& context,const WorkerInput& worker,const Matrix& knots,
    const GlobalParameters& parameters) {
  // M_d=A=(M-I)/d makes this block particularly compact: W A=(I-W)/d.
  // Consequently every occurrence of WAW or WAWAWQ reduces to W, W^2, or
  // W^3 actions.  No derivative matrix, W matrix, or Q matrix is materialized.
  const auto first=compute_device_delta_theta_delta_first_reduction_streamed(context,worker,knots,parameters);
  const auto state=build_device_mu_sigma_local(context,worker,knots,parameters);
  streamed_fixed_vectors(context,state);
  auto& r=context.region_; const double one=1.,zero=0.;
  // streamed_trace_inverse reuses summary[3] for tr(W); preserve the
  // validated delta score for the device-only final dd assembly.
  theta_copy<<<1,1,0,context.stream()>>>(r.delta_theta_summary+3,r.delta_theta_summary+9);
  const double trw=streamed_trace_inverse(context,state);
  // slots 4..10 are: tr(W^2), u'Wu, ||u-Wu||^2, tr(W^2Q), tr(W^3Q), and a tile scalar.
  theta_zero<<<1,1,0,context.stream()>>>(r.delta_theta_summary,4);
  // bk is n-by-m resident scratch: never issue a wider identity RHS tile.
  const int tile=std::min(128,state.m);
  for(int offset=0;offset<state.n;offset+=tile){const int cols=std::min(tile,state.n-offset);
    streamed_identity_tile<<<(state.n*cols+255)/256,256,0,context.stream()>>>(r.bk,state.n,cols,offset);
    check_dt(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,state.n,cols,&one,state.chol_M,state.n,r.bk,state.n));
    check_dt(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,state.n,cols,&one,state.chol_M,state.n,r.bk,state.n));
    streamed_square_tile<<<1,1,0,context.stream()>>>(r.bk,state.n,cols,r.delta_theta_summary+10);
    streamed_add_scalar<<<1,1,0,context.stream()>>>(r.delta_theta_summary+4,r.delta_theta_summary+10);
  }
  // rhs is a single reusable W-vector action.  It is overwritten after its
  // two scalar consumers, so its lifetime is O(n).
  check_dt(cudaMemcpyAsync(r.rhs,r.delta_theta_u,size_t(state.n)*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));
  check_dt(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,state.n,1,&one,state.chol_M,state.n,r.rhs,state.n));
  check_dt(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,state.n,1,&one,state.chol_M,state.n,r.rhs,state.n));
  streamed_dot_and_difference_norm<<<1,1,0,context.stream()>>>(r.delta_theta_u,r.rhs,state.n,r.delta_theta_summary+5,r.delta_theta_summary+6);
  // V=dWB.  V'V/d^2 is B'W^2B; after reducing it, overwrite bk with WV
  // and reduce V'(WV)/d^2 = B'W^3B.  Both are m*m summaries, not n*n state.
  check_dt(cublasDgemm(context.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,state.m,state.m,state.n,&one,state.minv_B,state.n,state.minv_B,state.n,&zero,r.bt_rinv_b,state.m));
  streamed_gram_trace_scaled<<<1,1,0,context.stream()>>>(r.bt_rinv_b,state.sigma,state.m,1./(parameters.delta*parameters.delta),r.delta_theta_summary+7);
  check_dt(cudaMemcpyAsync(r.bk,state.minv_B,size_t(state.n)*state.m*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));
  check_dt(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,state.n,state.m,&one,state.chol_M,state.n,r.bk,state.n));
  check_dt(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,state.n,state.m,&one,state.chol_M,state.n,r.bk,state.n));
  check_dt(cublasDgemm(context.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,state.m,state.m,state.n,&one,state.minv_B,state.n,r.bk,state.n,&zero,r.bt_rinv_b,state.m));
  streamed_gram_trace_scaled<<<1,1,0,context.stream()>>>(r.bt_rinv_b,state.sigma,state.m,1./(parameters.delta*parameters.delta),r.delta_theta_summary+8);
  const auto base=compute_device_delta_theta_base_reduction_streamed(context,worker,knots,parameters);
  if (device_only_components) return {};
  // A compact scalar download only; no model-sized derivative result crosses
  // the transport boundary.
  std::array<double,9> host{};
  check_dt(cudaMemcpyAsync(host.data()+4,r.delta_theta_summary+4,5*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));
  check_dt(cudaStreamSynchronize(context.stream())); r.result_d2h_bytes+=5*sizeof(double);
  const double d2=parameters.delta*parameters.delta;
  // u=W r.  The first-gradient identity gives u'u, but the first
  // derivative of r' W r needs r' W A W r = (r' W r-u'u)/delta.
  // In particular, u' W u belongs to the next (W A W A W) contraction;
  // substituting it here drops one power of W and is visible already at n=3.
  const double u2=first.delta_gradient+trw/parameters.delta-host[7];
  const double uau=(base.quadratic-u2)/parameters.delta;
  const double twawq=(base.trace_q-host[7])/parameters.delta;
  const double hd=-(uau+twawq);
  const double ldd=(2.*trw-host[4])/d2;
  // r' W A W A W r = (r'W r-2 r'W^2 r+r'W^3 r)/delta^2.
  // Here r'W^2r=u'u and r'W^3r=u'Wu.  The old ||u-Wu||^2 expression
  // instead carries W^4 and is not this contraction.
  const double hdd=2.*(base.quadratic-2.*u2+host[5]
                       +base.trace_q-2.*host[7]+host[8])/d2;
  return {first.objective,first.delta_gradient,ldd+2.*hd+parameters.delta*hdd,
          (double(state.n)-trw)/parameters.delta,
          (double(state.n)-2.*trw+host[4])/d2,
          uau,
          (base.trace_q-2.*host[7]+host[8])/d2};
}

DeviceDeltaThetaProducerDeltaDeltaAudit
compute_device_delta_theta_producer_delta_delta_audit_for_test(
    GpuContext& context,const WorkerInput& worker,const Matrix& knots,
    const GlobalParameters& parameters) {
  // Mirror the producer only through its delta/delta boundary.  This test
  // hook deliberately downloads after that boundary and before another
  // component can reuse summary slots.
  const auto standalone=compute_device_delta_theta_delta_hessian_reduction_streamed(
      context,worker,knots,parameters);
  DeviceOnlyComponentsScope device_only;
  (void)build_device_mu_sigma_local(context,worker,knots,parameters);
  auto& r=context.region_;
  if(!r.delta_theta_final) {
    check_dt(cudaMalloc(&r.delta_theta_final,13*sizeof(double)));
    ++r.delta_theta_reduction_allocation_count;
  }
  (void)compute_device_delta_theta_base_reduction_streamed(context,worker,knots,parameters);
  (void)compute_device_delta_theta_delta_first_reduction_streamed(context,worker,knots,parameters);
  theta_copy<<<1,1,0,context.stream()>>>(r.delta_theta_summary+3,r.delta_theta_final+1);
  (void)compute_device_delta_theta_delta_hessian_reduction_streamed(context,worker,knots,parameters);
  streamed_delta_hessian_from_components<<<1,1,0,context.stream()>>>(
      r.delta_theta_summary,parameters.delta,worker.sites.rows,r.delta_theta_final);
  std::array<double,13> snapshot{};
  check_dt(cudaMemcpyAsync(snapshot.data(),r.delta_theta_summary,13*sizeof(double),
                           cudaMemcpyDeviceToHost,context.stream()));
  double delta_score=0.,assembled=0.;
  check_dt(cudaMemcpyAsync(&delta_score,r.delta_theta_final+1,sizeof(double),
                           cudaMemcpyDeviceToHost,context.stream()));
  check_dt(cudaMemcpyAsync(&assembled,r.delta_theta_final+4,sizeof(double),
                           cudaMemcpyDeviceToHost,context.stream()));
  check_dt(cudaStreamSynchronize(context.stream()));
  r.result_d2h_bytes+=15*sizeof(double);
  return {standalone.delta_hessian,assembled,
          {snapshot[0],snapshot[1],delta_score,snapshot[3],snapshot[4],
           snapshot[5],snapshot[7],snapshot[8]}};
}

DeviceDeltaThetaMixedHessianReduction
compute_device_delta_theta_mixed_hessian_reduction_streamed(
    GpuContext& context,const WorkerInput& worker,const Matrix& knots,
    const GlobalParameters& parameters) {
  // Differentiate the already-streamed delta gradient
  //   -tr(W)/d + u'u + tr(W B Sigma B' W).
  // For T=M_theta this is
  // tr(W T W)/d + 2 r_theta' W u - 2 u' W T u
  // + 2 tr(B_theta Sigma B' W^2) - 2 tr(B Sigma B' W T W^2).
  // Every T use below is theta_mixed_action: one vector in, one vector out.
  const auto state=build_device_mu_sigma_local(context,worker,knots,parameters);
  streamed_fixed_vectors(context,state);
  auto& r=context.region_; const double one=1.,zero_scalar=0.;
  if(!r.delta_theta_summary) check_dt(cudaMalloc(&r.delta_theta_summary,13*sizeof(double)));
  DeviceDeltaThetaMixedHessianReduction result;
  for(int parameter=0;parameter<2;++parameter) {
    const char* parameter_filter=std::getenv("FEDSPATIAL_MIXED_PARAMETER");
    if(parameter_filter && std::atoi(parameter_filter)!=parameter) continue;
    const auto d=compute_device_b_pair_derivatives(context,parameters.nu2,
        parameters.alpha,parameters.range,parameter,parameter);
    const double* bi=d.first_i;
    auto t_action=[&](const double* in,double* out) {
      theta_a_action(context,state,bi,d.k_i,parameter,in,out);
      check_dt(cublasDscal(context.cublas(),state.n,&parameters.delta,out,1));
    };
    // term 0: tr(W T W)/d, traced by columns.  r.delta_theta_r and rhs are
    // reusable vectors, never a W/T product matrix.
    theta_zero<<<1,1,0,context.stream()>>>(r.delta_theta_summary,0);
    const int tile=producer_tile_width(state.m);
    for(int offset=0;offset<state.n;offset+=tile) {
      const int cols=std::min(tile,state.n-offset);
      streamed_identity_tile<<<(state.n*cols+255)/256,256,0,context.stream()>>>(r.cnk,state.n,cols,offset);
      theta_solve_tile(context,state,r.cnk,cols);
      if(mixed_trace_tile_enabled()) {
        theta_a_action_tile(context,state,bi,d.k_i,parameter,r.cnk,r.bk,cols);
        check_dt(cublasDscal(context.cublas(),state.n*cols,&parameters.delta,r.bk,1));
        theta_solve_tile(context,state,r.bk,cols);
        mixed_trace_tile_add<<<1,1,0,context.stream()>>>(r.bk,state.n,cols,offset,parameters.delta,r.delta_theta_summary);
        ++r.mixed_action_trace_tile_calls;
      } else for(int local=0;local<cols;++local) {
        const int col=offset+local; const double* wcol=r.cnk+size_t(local)*state.n;
        t_action(wcol,r.rhs);
        check_dt(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,state.n,1,&one,state.chol_M,state.n,r.rhs,state.n));
        check_dt(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,state.n,1,&one,state.chol_M,state.n,r.rhs,state.n));
        mixed_trace_add<<<1,1,0,context.stream()>>>(r.rhs,col,parameters.delta,r.delta_theta_summary);
      }
    }
    // terms 1 and 2: r_theta' W u and -u' W T u.
    check_dt(cudaMemcpyAsync(r.delta_theta_r,r.delta_theta_u,size_t(state.n)*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));
    check_dt(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,state.n,1,&one,state.chol_M,state.n,r.delta_theta_r,state.n));
    check_dt(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,state.n,1,&one,state.chol_M,state.n,r.delta_theta_r,state.n));
    t_action(r.delta_theta_u,r.rhs);
    check_dt(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,state.n,1,&one,state.chol_M,state.n,r.rhs,state.n));
    check_dt(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,state.n,1,&one,state.chol_M,state.n,r.rhs,state.n));
    mixed_vector_terms<<<1,1,0,context.stream()>>>(r.delta_theta_u,r.delta_theta_r,r.rhs,state.mu,bi,
        state.n,state.m,r.delta_theta_summary+1,r.delta_theta_summary+2);
    // Z=W V=d W^2 B.  This n-by-m state is reused column-by-column and is
    // not a derivative/W/Q n-by-n allocation.
    check_dt(cudaMemcpyAsync(r.bk,state.minv_B,size_t(state.n)*state.m*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));
    check_dt(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,state.n,state.m,&one,state.chol_M,state.n,r.bk,state.n));
    check_dt(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,state.n,state.m,&one,state.chol_M,state.n,r.bk,state.n));
    if(mixed_contractions_tile_enabled()) {
      // Bi is stored as a row-major n-by-m matrix, exactly the column-major
      // m-by-n view required by this GEMM.  bt_rinv_b is dead at this point
      // and is reused only until the following theta action reclaims it.
      check_dt(cublasDgemm(context.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,state.m,state.m,state.n,&one,
                           bi,state.m,r.bk,state.n,&zero_scalar,r.bt_rinv_b,state.m));
      mixed_bi_w2b_gram_add<<<1,1,0,context.stream()>>>(r.bt_rinv_b,state.sigma,state.m,parameters.delta,r.delta_theta_summary+3);
      ++r.mixed_bi_w2b_gemm_calls;
    } else mixed_bi_w2b<<<1,1,0,context.stream()>>>(bi,r.bk,state.sigma,state.n,state.m,parameters.delta,r.delta_theta_summary+3);
    theta_zero<<<1,1,0,context.stream()>>>(r.delta_theta_summary,4);
    if(mixed_contractions_tile_enabled()) {
      for(int offset=0;offset<state.m;offset+=tile) { const int cols=std::min(tile,state.m-offset);
        theta_a_action_tile(context,state,bi,d.k_i,parameter,r.bk+size_t(offset)*state.n,r.cnk,cols);
        check_dt(cublasDscal(context.cublas(),state.n*cols,&parameters.delta,r.cnk,1));
        // V' T(Z_tile): m-by-n times n-by-tile.  m0 is an owned tile scratch
        // released by theta_a_action_tile before this operation.
        check_dt(cublasDgemm(context.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,state.m,cols,state.n,&one,
                             state.minv_B,state.n,r.cnk,state.n,&zero_scalar,r.mixed_action_tile_m0,state.m));
        mixed_v_tz_tile_add<<<1,1,0,context.stream()>>>(r.mixed_action_tile_m0,state.sigma,state.m,offset,cols,parameters.delta,r.delta_theta_summary+4);
        ++r.mixed_v_tz_tile_calls; r.mixed_v_tz_rhs_columns+=cols;
      }
    } else for(int source=0;source<state.m;++source) {
      t_action(r.bk+size_t(state.n)*source,r.rhs);
      mixed_v_tz_add<<<1,1,0,context.stream()>>>(state.minv_B,r.rhs,state.sigma,source,state.n,state.m,parameters.delta,r.delta_theta_summary+4);
    }
    check_dt(cudaGetLastError());
    // The production summary keeps this scalar on device; the host record
    // below is retained solely for the focused component diagnostic API.
    if(r.delta_theta_final) {
      const int slot=parameter==0?5:6;
      theta_sum_copy<<<1,1,0,context.stream()>>>(r.delta_theta_summary,5,
                                                   r.delta_theta_final+slot);
      theta_copy<<<1,1,0,context.stream()>>>(r.delta_theta_final+slot,
                                               r.delta_theta_final+(parameter==0?7:10));
    }
    if (device_only_components) continue;
    std::array<double,5> host{};
    check_dt(cudaMemcpyAsync(host.data(),r.delta_theta_summary,5*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));
    check_dt(cudaStreamSynchronize(context.stream())); r.result_d2h_bytes+=5*sizeof(double);
    result.terms[parameter]=host;
    for(double value:host) result.mixed[parameter]+=value;
  }
  return result;
}

DeviceDeltaThetaThetaHessianReduction
compute_device_delta_theta_theta_hessian_reduction_streamed(
    GpuContext& context,const WorkerInput& worker,const Matrix& knots,
    const GlobalParameters& parameters) {
  // For i,j in {alpha,range}, differentiate the exact theta score:
  // tr(W M_i) + delta*(q_i+t_i).  The three groups below are
  // tr(W M_ij)-tr(W M_j W M_i), delta*q_ij, and delta*t_ij.
  // Every matrix occurrence is an action on one vector; each entry owns only
  // the reusable B_i/B_j/B_ij pair workspace and its three m-by-m K actions.
  const auto s=build_device_mu_sigma_local(context,worker,knots,parameters);
  streamed_fixed_vectors(context,s);
  auto&r=context.region_; if(!r.delta_theta_summary)check_dt(cudaMalloc(&r.delta_theta_summary,13*sizeof(double)));
  const double one=1.;
  auto zero=[&](double*x){theta_zero_vector<<<(s.n+255)/256,256,0,context.stream()>>>(x,s.n);};
  auto adddot=[&](const double*x,const double*y,double scale,int slot){theta_dot_add<<<1,1,0,context.stream()>>>(x,y,s.n,scale,r.delta_theta_summary+slot);};
  auto addcol=[&](const double*x,int col,double scale,int slot){theta_column_add<<<1,1,0,context.stream()>>>(x,col,scale,r.delta_theta_summary+slot);};
  DeviceDeltaThetaThetaHessianReduction out;
  const int pp[3][2]={{0,0},{0,1},{1,1}};
  for(int entry=0;entry<3;++entry){const char* entry_filter=std::getenv("FEDSPATIAL_THETA_HESSIAN_ENTRY");if(entry_filter&&std::atoi(entry_filter)!=entry)continue;const int i=pp[entry][0],j=pp[entry][1];const auto d=compute_device_b_pair_derivatives(context,parameters.nu2,parameters.alpha,parameters.range,i,j);auto m1=[&](int p,const double*x,double*y){const bool left=p==i;theta_a_action(context,s,left?d.first_i:d.first_j,left?d.k_i:d.k_j,p,x,y);check_dt(cublasDscal(context.cublas(),s.n,&parameters.delta,y,1));};auto m2=[&](const double*x,double*y){theta_a_second_action(context,s,d,i,j,x,y);check_dt(cublasDscal(context.cublas(),s.n,&parameters.delta,y,1));};auto q1=[&](int p,const double*x,double*y){const double*b=p==i?d.first_i:d.first_j;theta_q_action(context,b,r.b,x,y);theta_triple_action(context,r.b,r.parameter_sigma,b,x,y,1.);};auto ri=[&](int p,double*y){const double*b=p==i?d.first_i:d.first_j;const double z=0.;check_dt(cublasDgemv(context.cublas(),CUBLAS_OP_T,s.m,s.n,&one,b,s.m,s.mu,1,&z,y,1));};auto rij=[&](double*y){const double z=0.;check_dt(cublasDgemv(context.cublas(),CUBLAS_OP_T,s.m,s.n,&one,d.second,s.m,s.mu,1,&z,y,1));};theta_zero<<<1,1,0,context.stream()>>>(r.delta_theta_summary,0);theta_zero<<<1,1,0,context.stream()>>>(r.delta_theta_summary,1);theta_zero<<<1,1,0,context.stream()>>>(r.delta_theta_summary,2);
    // Log determinant columns are solved together but consumed in the same
    // canonical increasing-column order as the scalar path.
    const int tile=producer_tile_width(s.m);
    for(int offset=0;offset<s.n;offset+=tile){const int cols=std::min(tile,s.n-offset);
      streamed_identity_tile<<<(s.n*cols+255)/256,256,0,context.stream()>>>(r.cnk,s.n,cols,offset);
      theta_solve_tile(context,s,r.cnk,cols);
      const bool tile_log=(entry==0&&theta_hessian_aa_log_tile_enabled())||(entry==1&&theta_hessian_ar_log_tile_enabled())||(entry==2&&theta_hessian_rr_log_tile_enabled());
      if(tile_log) {
        // Retain immediate scalar consumption.  Both the GEMM candidate and
        // deferred-output launch prototype fail the ill-conditioned parity
        // gate; this preserves the canonical action/scratch lifetime.
        if(theta_hessian_second_log_tile_enabled()) {
          // Opt-in oracle candidate: preserve the fixed ascending diagonal
          // reduction while batching only independent M_ij actions.
          theta_a_second_action_tile(context,s,d,i,j,r.cnk,r.bk,cols);
          check_dt(cublasDscal(context.cublas(),s.n*cols,&parameters.delta,r.bk,1));
          theta_diag_tile_add<<<1,1,0,context.stream()>>>(r.bk,s.n,cols,offset,1.,r.delta_theta_summary);
        } else for(int local=0;local<cols;++local){const int col=offset+local;const double*wcol=r.cnk+size_t(local)*s.n;m2(wcol,r.rhs);addcol(r.rhs,col,1.,0);}
        theta_a_action_tile(context,s,d.first_i,d.k_i,i,r.cnk,r.bk,cols);
        check_dt(cublasDscal(context.cublas(),s.n*cols,&parameters.delta,r.bk,1));
        theta_solve_tile(context,s,r.bk,cols);
        theta_a_action_tile(context,s,d.first_j,d.k_j,j,r.bk,r.cnk,cols);
        check_dt(cublasDscal(context.cublas(),s.n*cols,&parameters.delta,r.cnk,1));
        theta_diag_tile_add<<<1,1,0,context.stream()>>>(r.cnk,s.n,cols,offset,-1.,r.delta_theta_summary);
        if(entry==0){++r.theta_hessian_aa_log_tile_calls;r.theta_hessian_aa_log_rhs_columns+=cols;}
        else if(entry==1){++r.theta_hessian_ar_log_tile_calls;r.theta_hessian_ar_log_rhs_columns+=cols;}
        else {++r.theta_hessian_rr_log_tile_calls;r.theta_hessian_rr_log_rhs_columns+=cols;}
      } else for(int local=0;local<cols;++local){const int col=offset+local;const double*wcol=r.cnk+size_t(local)*s.n;
          m2(wcol,r.rhs);addcol(r.rhs,col,1.,0);m1(i,wcol,r.rhs);theta_solve(context,s,r.rhs);m1(j,r.rhs,r.bk);addcol(r.bk,col,-1.,0);}}
    // residual quadratic group.
    rij(r.delta_theta_r);adddot(r.delta_theta_r,r.delta_theta_u,2.*parameters.delta,1);
    ri(j,r.delta_theta_r);theta_solve(context,s,r.delta_theta_r);ri(i,r.rhs);adddot(r.rhs,r.delta_theta_r,2.*parameters.delta,1);
    m1(j,r.delta_theta_u,r.delta_theta_r);theta_solve(context,s,r.delta_theta_r);ri(i,r.rhs);adddot(r.rhs,r.delta_theta_r,-2.*parameters.delta,1);
    m1(i,r.delta_theta_u,r.delta_theta_r);theta_solve(context,s,r.delta_theta_r);ri(j,r.rhs);adddot(r.rhs,r.delta_theta_r,-2.*parameters.delta,1);
    m1(i,r.delta_theta_u,r.delta_theta_r);theta_solve(context,s,r.delta_theta_r);m1(j,r.delta_theta_r,r.rhs);adddot(r.delta_theta_u,r.rhs,parameters.delta,1);
    m1(j,r.delta_theta_u,r.delta_theta_r);theta_solve(context,s,r.delta_theta_r);m1(i,r.delta_theta_r,r.rhs);adddot(r.delta_theta_u,r.rhs,parameters.delta,1);
    m2(r.delta_theta_u,r.rhs);adddot(r.delta_theta_u,r.rhs,-parameters.delta,1);
    // A W column is used six times below.  Batch its independent solves once,
    // then retain the original per-column/per-term reduction sequence.
    for(int offset=0;offset<s.n;offset+=tile){const int cols=std::min(tile,s.n-offset);
      streamed_identity_tile<<<(s.n*cols+255)/256,256,0,context.stream()>>>(r.cnk,s.n,cols,offset);
      theta_solve_tile(context,s,r.cnk,cols);
      if(theta_hessian_q_w_tile_enabled()) {
        auto m1_tile=[&](int p){const bool left=p==i;theta_a_action_tile(context,s,left?d.first_i:d.first_j,left?d.k_i:d.k_j,p,r.bk,r.bk,cols);check_dt(cublasDscal(context.cublas(),s.n*cols,&parameters.delta,r.bk,1));};
        const bool q_panel=theta_hessian_q_panel_tile_enabled();
        if(q_panel && theta_hessian_q_superpanel_enabled()) {
          theta_q_second_first_superpanel(context,s,d,r.cnk,cols);
          const size_t out_stride=size_t(s.n)*tile;
          check_dt(cudaMemcpyAsync(r.bk,r.theta_q_super_out,out_stride*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));
          ++r.theta_q_super_output_copies;
          theta_diag_tile_add<<<1,1,0,context.stream()>>>(r.bk,s.n,cols,offset,parameters.delta,r.delta_theta_summary+2);
          check_dt(cudaMemcpyAsync(r.bk,r.theta_q_super_out+out_stride,out_stride*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));
          ++r.theta_q_super_output_copies;
          theta_solve_tile(context,s,r.bk,cols);m1_tile(j);theta_diag_tile_add<<<1,1,0,context.stream()>>>(r.bk,s.n,cols,offset,-parameters.delta,r.delta_theta_summary+2);
          check_dt(cudaMemcpyAsync(r.bk,r.theta_q_super_out+2*out_stride,out_stride*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));
          ++r.theta_q_super_output_copies;
          theta_solve_tile(context,s,r.bk,cols);m1_tile(i);theta_diag_tile_add<<<1,1,0,context.stream()>>>(r.bk,s.n,cols,offset,-parameters.delta,r.delta_theta_summary+2);
          theta_q_super_check_guards(context);
        } else {
          if(q_panel) { theta_q_second_action_tile(context,s,d,r.cnk,r.bk,cols); theta_diag_tile_add<<<1,1,0,context.stream()>>>(r.bk,s.n,cols,offset,parameters.delta,r.delta_theta_summary+2); }
          else for(int local=0;local<cols;++local){const int col=offset+local;theta_q_second_action(context,d,r.cnk+size_t(local)*s.n,r.rhs);addcol(r.rhs,col,parameters.delta,2);}
          if(q_panel) theta_q_first_action_tile(context,s,d.first_i,r.cnk,r.bk,cols);
          else for(int local=0;local<cols;++local)q1(i,r.cnk+size_t(local)*s.n,r.bk+size_t(local)*s.n);
          theta_solve_tile(context,s,r.bk,cols);m1_tile(j);theta_diag_tile_add<<<1,1,0,context.stream()>>>(r.bk,s.n,cols,offset,-parameters.delta,r.delta_theta_summary+2);
          if(q_panel) theta_q_first_action_tile(context,s,d.first_j,r.cnk,r.bk,cols);
          else for(int local=0;local<cols;++local)q1(j,r.cnk+size_t(local)*s.n,r.bk+size_t(local)*s.n);
          theta_solve_tile(context,s,r.bk,cols);m1_tile(i);theta_diag_tile_add<<<1,1,0,context.stream()>>>(r.bk,s.n,cols,offset,-parameters.delta,r.delta_theta_summary+2);
        }
        if(q_panel && theta_hessian_q_base_reuse_enabled()) {
          // Q=B Sigma B' W is algebraically identical for the three residual
          // chains.  Materialize it once in a dedicated cache, then retain
          // the original branch-local GEMMs/TRSMs and terminal column order.
          // The cache stores WQ after one solve; copying it back to bk before
          // each mutable branch avoids changing either dot-product order or
          // an action's in/out scratch lifetime.
          theta_q_base_cache_ensure(context,size_t(s.n)*tile);
          r.theta_q_base_logical_actions+=3;
          r.theta_q_base_logical_solves+=5;
          theta_q_action_tile(context,s,r.b,r.b,r.cnk,r.theta_q_base_cache,cols);
          ++r.theta_q_base_actual_actions;
          theta_solve_tile(context,s,r.theta_q_base_cache,cols);
          ++r.theta_q_base_actual_solves;
          check_dt(cudaMemcpyAsync(r.bk,r.theta_q_base_cache,size_t(s.n)*cols*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));
          ++r.theta_q_base_cache_copies;
          m1_tile(i); theta_solve_tile(context,s,r.bk,cols); ++r.theta_q_base_actual_solves;
          m1_tile(j); theta_diag_tile_add<<<1,1,0,context.stream()>>>(r.bk,s.n,cols,offset,parameters.delta,r.delta_theta_summary+2);
          check_dt(cudaMemcpyAsync(r.bk,r.theta_q_base_cache,size_t(s.n)*cols*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));
          ++r.theta_q_base_cache_copies;
          m1_tile(j); theta_solve_tile(context,s,r.bk,cols); ++r.theta_q_base_actual_solves;
          m1_tile(i); theta_diag_tile_add<<<1,1,0,context.stream()>>>(r.bk,s.n,cols,offset,parameters.delta,r.delta_theta_summary+2);
          theta_a_second_action_tile(context,s,d,i,j,r.theta_q_base_cache,r.cnk,cols);
          check_dt(cublasDscal(context.cublas(),s.n*cols,&parameters.delta,r.cnk,1));
          theta_diag_tile_add<<<1,1,0,context.stream()>>>(r.cnk,s.n,cols,offset,-parameters.delta,r.delta_theta_summary+2);
          theta_q_base_cache_check_guards(context);
        } else {
          if(q_panel) theta_q_action_tile(context,s,r.b,r.b,r.cnk,r.bk,cols);
          else for(int local=0;local<cols;++local)theta_q_action(context,r.b,r.b,r.cnk+size_t(local)*s.n,r.bk+size_t(local)*s.n);
          theta_solve_tile(context,s,r.bk,cols);m1_tile(i);theta_solve_tile(context,s,r.bk,cols);m1_tile(j);theta_diag_tile_add<<<1,1,0,context.stream()>>>(r.bk,s.n,cols,offset,parameters.delta,r.delta_theta_summary+2);
          if(q_panel) theta_q_action_tile(context,s,r.b,r.b,r.cnk,r.bk,cols);
          else for(int local=0;local<cols;++local)theta_q_action(context,r.b,r.b,r.cnk+size_t(local)*s.n,r.bk+size_t(local)*s.n);
          theta_solve_tile(context,s,r.bk,cols);m1_tile(j);theta_solve_tile(context,s,r.bk,cols);m1_tile(i);theta_diag_tile_add<<<1,1,0,context.stream()>>>(r.bk,s.n,cols,offset,parameters.delta,r.delta_theta_summary+2);
          if(q_panel) theta_q_action_tile(context,s,r.b,r.b,r.cnk,r.bk,cols);
          else for(int local=0;local<cols;++local)theta_q_action(context,r.b,r.b,r.cnk+size_t(local)*s.n,r.bk+size_t(local)*s.n);
          theta_solve_tile(context,s,r.bk,cols);
          if(q_panel) { theta_a_second_action_tile(context,s,d,i,j,r.bk,r.cnk,cols); check_dt(cublasDscal(context.cublas(),s.n*cols,&parameters.delta,r.cnk,1)); theta_diag_tile_add<<<1,1,0,context.stream()>>>(r.cnk,s.n,cols,offset,-parameters.delta,r.delta_theta_summary+2); }
          else for(int local=0;local<cols;++local){const int col=offset+local;m2(r.bk+size_t(local)*s.n,r.rhs);addcol(r.rhs,col,-parameters.delta,2);}
        }
        r.theta_hessian_q_tile_calls+=5;r.theta_hessian_q_rhs_columns+=5*cols;
      } else for(int local=0;local<cols;++local){const int col=offset+local;const double*wcol=r.cnk+size_t(local)*s.n;
        theta_q_second_action(context,d,wcol,r.rhs);addcol(r.rhs,col,parameters.delta,2);
        q1(i,wcol,r.rhs);theta_solve(context,s,r.rhs);m1(j,r.rhs,r.bk);addcol(r.bk,col,-parameters.delta,2);
        q1(j,wcol,r.rhs);theta_solve(context,s,r.rhs);m1(i,r.rhs,r.bk);addcol(r.bk,col,-parameters.delta,2);
        theta_q_action(context,r.b,r.b,wcol,r.rhs);theta_solve(context,s,r.rhs);m1(i,r.rhs,r.delta_theta_r);theta_solve(context,s,r.delta_theta_r);m1(j,r.delta_theta_r,r.bk);addcol(r.bk,col,parameters.delta,2);
        theta_q_action(context,r.b,r.b,wcol,r.rhs);theta_solve(context,s,r.rhs);m1(j,r.rhs,r.delta_theta_r);theta_solve(context,s,r.delta_theta_r);m1(i,r.delta_theta_r,r.bk);addcol(r.bk,col,parameters.delta,2);
        theta_q_action(context,r.b,r.b,wcol,r.rhs);theta_solve(context,s,r.rhs);m2(r.rhs,r.bk);addcol(r.bk,col,-parameters.delta,2);
      }}
    if(r.delta_theta_final) {
      const int slot=entry==0?8:(entry==1?9:12);
      theta_sum_copy<<<1,1,0,context.stream()>>>(r.delta_theta_summary,3,
                                                   r.delta_theta_final+slot);
      if(entry==1) theta_copy<<<1,1,0,context.stream()>>>(r.delta_theta_final+slot,
                                                            r.delta_theta_final+11);
    }
    if (!device_only_components) { std::array<double,3> h{};check_dt(cudaMemcpyAsync(h.data(),r.delta_theta_summary,3*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check_dt(cudaStreamSynchronize(context.stream()));r.result_d2h_bytes+=3*sizeof(double);out.terms[entry]=h;out.hessian[entry]=h[0]+h[1]+h[2]; }
  }
  return out;
}

DeviceDeltaThetaThetaFirstReduction
compute_device_delta_theta_theta_first_reduction_streamed(
    GpuContext& context,const WorkerInput& worker,const Matrix& knots,
    const GlobalParameters& parameters) {
  // This path is intentionally independent of build_device_delta_theta_*:
  // those compatibility builders retain W/Q and full first/second bundles.
  const auto state=build_device_mu_sigma_local(context,worker,knots,parameters);
  streamed_fixed_vectors(context,state);
  auto& r=context.region_; const double one=1.,zero=0.;
  if(!r.delta_theta_summary) check_dt(cudaMalloc(&r.delta_theta_summary,13*sizeof(double)));
  theta_base_finish<<<1,1,0,context.stream()>>>(state.chol_M,r.delta_theta_r,r.delta_theta_u,
      state.bt_minv_B,state.sigma,parameters.delta,state.n,state.m,r.delta_theta_summary);
  check_dt(cudaGetLastError());
  // delta_theta_r is now a reusable single inverse column.  It is never a W
  // matrix: each column is solved, consumed in increasing index order, then
  // overwritten.  bk is likewise a single n-vector output for M_i V[:,j].
  for(int parameter=0;parameter<2;++parameter) {
    const char* parameter_filter=std::getenv("FEDSPATIAL_THETA_FIRST_PARAMETER");
    if(parameter_filter && std::atoi(parameter_filter)!=parameter) continue;
    const auto bi=compute_device_b_first_derivative(context,parameters.nu2,
        parameters.alpha,parameters.range,parameter);
    auto m_theta_action = [&](const double* input, double* output) {
      theta_a_action(context,state,bi.derivative,r.k_first,parameter,input,output);
      check_dt(cublasDscal(context.cublas(),state.n,&parameters.delta,output,1));
    };
    const int slot=2+parameter;
    theta_zero<<<1,1,0,context.stream()>>>(r.delta_theta_summary,slot);
    const int tile=producer_tile_width(state.m);
    for(int offset=0;offset<state.n;offset+=tile) {
      const int cols=std::min(tile,state.n-offset);
      streamed_identity_tile<<<(state.n*cols+255)/256,256,0,context.stream()>>>(r.cnk,state.n,cols,offset);
      theta_solve_tile(context,state,r.cnk,cols);
      if(theta_first_tile_enabled()) {
        streamed_identity_tile<<<(state.n*cols+255)/256,256,0,context.stream()>>>(r.bk,state.n,cols,offset);
        // Keep W in cnk.  The action first copies the identity tile to its
        // owned input, so bk can safely be reused as its output afterwards.
        theta_a_action_tile(context,state,bi.derivative,r.k_first,parameter,r.bk,r.bk,cols);
        check_dt(cublasDscal(context.cublas(),state.n*cols,&parameters.delta,r.bk,1));
        theta_trace_tile_dot<<<1,1,0,context.stream()>>>(r.cnk,r.bk,state.n,cols,offset,r.delta_theta_summary+slot);
        ++r.theta_first_trace_tile_calls; r.theta_first_trace_rhs_columns+=cols;
      } else for(int local=0;local<cols;++local) {
          const int col=offset+local; const double* wcol=r.cnk+size_t(local)*state.n;
          streamed_identity_tile<<<(state.n+255)/256,256,0,context.stream()>>>(r.rhs,state.n,1,col);
          m_theta_action(r.rhs,r.bk);
          theta_dot_add<<<1,1,0,context.stream()>>>(wcol,r.bk,state.n,1.,r.delta_theta_summary+slot);
      }
    }
    theta_copy<<<1,1,0,context.stream()>>>(r.delta_theta_summary+slot,
                                             r.delta_theta_summary+(parameter ? 12 : 1));
    // Keep the scalar terms parameter-private, both to avoid a scratch alias
    // and to make the dense identity auditable without retaining n-by-n data.
    const int terms=4+4*parameter;
    theta_ri_u<<<1,1,0,context.stream()>>>(r.delta_theta_u,state.mu,bi.derivative,
        state.n,state.m,r.delta_theta_summary+terms);
    theta_zero<<<1,1,0,context.stream()>>>(r.delta_theta_summary,terms+1);
    m_theta_action(r.delta_theta_u,r.bk);
    theta_dot_add<<<1,1,0,context.stream()>>>(r.delta_theta_u,r.bk,state.n,1.,
                                                r.delta_theta_summary+terms+1);
    if(theta_first_tile_enabled()) {
      check_dt(cublasDgemm(context.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,state.m,state.m,state.n,&one,
                           bi.derivative,state.m,state.minv_B,state.n,&zero,r.bt_rinv_b,state.m));
      theta_t1_gram<<<1,1,0,context.stream()>>>(r.bt_rinv_b,state.sigma,state.m,parameters.delta,r.delta_theta_summary+terms+2);
      ++r.theta_first_t1_gemm_calls;
    } else theta_t1<<<1,1,0,context.stream()>>>(bi.derivative,state.minv_B,state.sigma,state.n,state.m,
                                                   parameters.delta,r.delta_theta_summary+terms+2);
    theta_zero<<<1,1,0,context.stream()>>>(r.delta_theta_summary,terms+3);
    if(theta_first_tile_enabled()) {
      for(int offset=0;offset<state.m;offset+=tile){const int cols=std::min(tile,state.m-offset);
        theta_a_action_tile(context,state,bi.derivative,r.k_first,parameter,state.minv_B+size_t(offset)*state.n,r.cnk,cols);
        check_dt(cublasDscal(context.cublas(),state.n*cols,&parameters.delta,r.cnk,1));
        check_dt(cublasDgemm(context.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,state.m,cols,state.n,&one,
                             state.minv_B,state.n,r.cnk,state.n,&zero,r.mixed_action_tile_m0,state.m));
        theta_t2_tile_add<<<1,1,0,context.stream()>>>(r.mixed_action_tile_m0,state.sigma,state.m,offset,cols,r.delta_theta_summary+terms+3);
        ++r.theta_first_t2_tile_calls;r.theta_first_t2_rhs_columns+=cols;
      }
    } else for(int source=0;source<state.m;++source) {
        m_theta_action(state.minv_B+size_t(source)*state.n,r.bk);
        theta_t2_column<<<1,1,0,context.stream()>>>(state.minv_B,r.bk,state.sigma,source,state.n,
            state.m,r.delta_theta_summary+terms+3);
    }
    // V=delta*W*B and M_i=delta*A_i.  Therefore V' M_i V is
    // delta^2 times tr(W M_i W Q), not delta times it.
    theta_scale<<<1,1,0,context.stream()>>>(r.delta_theta_summary+terms+3,
                                              1./(parameters.delta*parameters.delta));
    theta_finish<<<1,1,0,context.stream()>>>(r.delta_theta_summary+terms,r.delta_theta_summary+terms+1,
        r.delta_theta_summary+terms+2,r.delta_theta_summary+terms+3,parameters.delta,slot,r.delta_theta_summary);
  }
  check_dt(cudaGetLastError());
  if (device_only_components) return {};
  std::array<double,4> host{};
  check_dt(cudaMemcpyAsync(host.data(),r.delta_theta_summary,4*sizeof(double),cudaMemcpyDeviceToHost,
                           context.stream()));
  check_dt(cudaStreamSynchronize(context.stream())); r.result_d2h_bytes+=4*sizeof(double);
  return {host[0],host[2],host[3]};
}

DeviceDeltaThetaThetaFirstDiagnostics
compute_device_delta_theta_theta_first_diagnostics_for_test(
    GpuContext& context,const WorkerInput& worker,const Matrix& knots,
    const GlobalParameters& parameters) {
  (void)compute_device_delta_theta_theta_first_reduction_streamed(context,worker,knots,parameters);
  std::array<double,13> host{}; auto& r=context.region_;
  check_dt(cudaMemcpyAsync(host.data(),r.delta_theta_summary,13*sizeof(double),cudaMemcpyDeviceToHost,
                           context.stream()));
  check_dt(cudaStreamSynchronize(context.stream())); r.result_d2h_bytes+=13*sizeof(double);
  DeviceDeltaThetaThetaFirstDiagnostics out;
  for(int parameter=0;parameter<2;++parameter) { const int terms=4+4*parameter;
    out.values[parameter]={host[parameter ? 12 : 1],host[terms],host[terms+1],host[terms+2],host[terms+3],host[2+parameter]}; }
  return out;
}

DeviceDeltaThetaSummary compute_device_delta_theta_summary(
    GpuContext& context,const WorkerInput& worker,const Matrix& knots,const GlobalParameters& parameters) {
  // The component paths below are the parity-clean streamed contractions.
  // Their compact scalar record is reused in-place: the final 13 values are
  // copied/summed device-to-device, never assembled from host results.  In
  // particular this deliberately does not enter build_device_delta_theta_*
  // or allocate the former 25*n*n reduction arena.
  DeviceOnlyComponentsScope device_only;
  // Bind before installing the producer cache scope.  bind_region resets a
  // cold RegionGpuState; installing the scope first used to erase its active
  // flag on the first producer call, giving cold contexts a different (and
  // observably non-repeatable) execution path from warm ones.
  context.bind_region(worker,knots);
  ProducerStateCacheScope state_cache(context);
  (void)build_device_mu_sigma_local(context,worker,knots,parameters);
  auto& region=context.region_;
  // Transport and component results must never alias: every reducer reuses
  // delta_theta_summary, and the mixed/theta-Hessian reducers retain several
  // scalars after their last valid write.  Preserve each completed stage in
  // the final 13-double record before starting the next one.
  if(!region.delta_theta_final) {
    check_dt(cudaMalloc(&region.delta_theta_final,13*sizeof(double)));
    region.delta_theta_final_capacity=13;
    ++region.delta_theta_reduction_allocation_count;
  }
  if(!region.delta_theta_component_scratch) {
    check_dt(cudaMalloc(&region.delta_theta_component_scratch,13*sizeof(double)));
    region.delta_theta_component_scratch_capacity=13;
  }
  double* final=region.delta_theta_final;
  const char* substage_env=std::getenv("FEDSPATIAL_DELTA_THETA_SUBSTAGE_TIMING");
  const bool substage_timing=substage_env && substage_env[0] && substage_env[0]!='0';
  // This checkpointed event helper is deliberately opt-in.  The profiling
  // mode measures stream work at component boundaries; production keeps the
  // original asynchronous sequence and only synchronizes at its transport
  // boundary.
  auto timed=[&](const char* name,auto&& work) {
    if(!substage_timing) { work(); return; }
    cudaEvent_t begin{},end{}; check_dt(cudaEventCreate(&begin)); check_dt(cudaEventCreate(&end));
    check_dt(cudaEventRecord(begin,context.stream())); work(); check_dt(cudaEventRecord(end,context.stream()));
    check_dt(cudaEventSynchronize(end)); float milliseconds=0.f; check_dt(cudaEventElapsedTime(&milliseconds,begin,end));
    std::lock_guard<std::mutex> lock(producer_cache_log_mutex);
    std::cerr<<"DELTA_THETA_SUBSTAGE device="<<context.device_id()<<" name="<<name
             <<" cuda_ms="<<milliseconds<<std::endl;
    check_dt(cudaEventDestroy(begin)); check_dt(cudaEventDestroy(end));
  };
  // Legacy transport layout (row-major, symmetric Hessian):
  // 0 objective; 1 d_delta, 2 d_alpha, 3 d_range;
  // 4 dd, 5 da, 6 dr, 7 ad, 8 aa, 9 ar, 10 rd, 11 ra, 12 rr.
  // Source scratch / last-valid point by stage:
  // base: summary[2] after base; delta-first: summary[3] after delta-first;
  // delta-Hessian: summary[0,1,3,4,5,7,8] after delta-Hessian;
  // mixed: summary[0..4] after each parameter; theta-first: summary[2,3]
  // after theta-first; theta-Hessian: summary[0..2] after each entry.
  timed("base",[&]{(void)compute_device_delta_theta_base_reduction_streamed(context,worker,knots,parameters);});
  theta_copy<<<1,1,0,context.stream_>>>(region.delta_theta_summary+2,final);
  capture_producer_boundary(context,0);
  timed("delta_first",[&]{(void)compute_device_delta_theta_delta_first_reduction_streamed(context,worker,knots,parameters);});
  theta_copy<<<1,1,0,context.stream_>>>(region.delta_theta_summary+3,final+1);
  capture_producer_boundary(context,1);
  timed("delta_hessian",[&]{(void)compute_device_delta_theta_delta_hessian_reduction_streamed(context,worker,knots,parameters);});
  // This is the exact streamed delta-Hessian source record, before its
  // 13-value transport slot is assembled.
  capture_producer_boundary(context,2);
  // The public delta-Hessian diagnostic returns a host value.  Its exact
  // scalar ingredients are already in component scratch, so production
  // emits the transport entry directly on the device.
  streamed_delta_hessian_from_components<<<1,1,0,context.stream_>>>(
      region.delta_theta_summary, parameters.delta, worker.sites.rows, final);
  // summary[4] is tr(W^2), an input to the preceding assembly rather than
  // the completed H(delta,delta) value.  Do not overwrite final[4].
  capture_producer_boundary(context,3);
  timed("mixed_alpha_range",[&]{(void)compute_device_delta_theta_mixed_hessian_reduction_streamed(context,worker,knots,parameters);});
  timed("theta_first_alpha_range",[&]{(void)compute_device_delta_theta_theta_first_reduction_streamed(context,worker,knots,parameters);});
  theta_copy<<<1,1,0,context.stream_>>>(region.delta_theta_summary+2,final+2);
  theta_copy<<<1,1,0,context.stream_>>>(region.delta_theta_summary+3,final+3);
  timed("theta_hessian_aa_ar_rr",[&]{(void)compute_device_delta_theta_theta_hessian_reduction_streamed(context,worker,knots,parameters);});
  check_dt(cudaGetLastError());
  return {final};
}

DeviceDeltaThetaEvaluation compute_device_delta_theta_local(
    GpuContext& context,const WorkerInput& worker,const Matrix& knots,const GlobalParameters& parameters) {
  const auto summary=compute_device_delta_theta_summary(context,worker,knots,parameters);
  std::array<double,13> host{}; auto& region=context.region_;
  check_dt(cudaMemcpyAsync(host.data(),summary.values,13*sizeof(double),cudaMemcpyDeviceToHost,context.stream_));
  check_dt(cudaStreamSynchronize(context.stream_)); region.result_d2h_bytes+=13*sizeof(double);
  DeviceDeltaThetaEvaluation out;out.summary.objective=host[0];for(int i=0;i<3;++i)out.summary.gradient[i]=host[1+i];for(int i=0;i<9;++i)out.summary.hessian[i]=host[4+i];return out;
}

DeviceDeltaThetaProducerBoundaryAudit compute_device_delta_theta_producer_boundary_audit_for_test(
    GpuContext& context,const WorkerInput& worker,const Matrix& knots,const GlobalParameters& parameters) {
  DeviceDeltaThetaProducerBoundaryAudit audit{};
  if(producer_boundary_audit) throw std::runtime_error("nested producer boundary audit");
  producer_boundary_audit=&audit;
  try { (void)compute_device_delta_theta_summary(context,worker,knots,parameters); }
  catch(...) { producer_boundary_audit=nullptr; throw; }
  producer_boundary_audit=nullptr;
  return audit;
}

DeltaThetaLocalResult compute_delta_theta_local(GpuContext& context,const WorkerInput& worker,const Matrix& knots,const GlobalParameters& parameters,const PGlobalDerivativeWorkspace&) { return compute_device_delta_theta_local(context,worker,knots,parameters).summary; }
DeltaThetaLocalResult compute_delta_theta_local(GpuContext& context,const WorkerInput& worker,const Matrix& knots,const GlobalParameters& parameters) { return compute_device_delta_theta_local(context,worker,knots,parameters).summary; }

Matrix test_theta_a_action_tile(GpuContext& context,const WorkerInput& worker,const Matrix& knots,const GlobalParameters& parameters,int parameter,const Matrix& rhs,int tile_width,bool tiled) {
  if(parameter<0||parameter>1||rhs.rows!=worker.sites.rows||tile_width<1)throw std::runtime_error("invalid theta action tile test input");
  const auto state=build_device_mu_sigma_local(context,worker,knots,parameters);
  const auto d=compute_device_b_pair_derivatives(context,parameters.nu2,parameters.alpha,parameters.range,parameter,parameter);
  // Deliberately leave a device canary at each end of both test buffers.  This
  // wrapper is only an oracle for the tile primitive, so it is worth making
  // out-of-range writes fail here instead of relying on a later producer test.
  const int n=state.n,columns=rhs.cols; const size_t count=size_t(n)*columns;
  constexpr double guard0=0x1.23456789abcp+17,guard1=-0x1.987654321fedcp-19;
  double *in_raw{},*out_raw{};check_dt(cudaMalloc(&in_raw,(count+2)*sizeof(double)));check_dt(cudaMalloc(&out_raw,(count+2)*sizeof(double)));
  double* in=in_raw+1; double* out=out_raw+1;
  check_dt(cudaMemcpyAsync(in_raw,&guard0,sizeof(double),cudaMemcpyHostToDevice,context.stream()));
  check_dt(cudaMemcpyAsync(in_raw+count+1,&guard1,sizeof(double),cudaMemcpyHostToDevice,context.stream()));
  check_dt(cudaMemcpyAsync(out_raw,&guard0,sizeof(double),cudaMemcpyHostToDevice,context.stream()));
  check_dt(cudaMemcpyAsync(out_raw+count+1,&guard1,sizeof(double),cudaMemcpyHostToDevice,context.stream()));
  check_dt(cudaMemcpyAsync(in,rhs.a.data(),size_t(n)*columns*sizeof(double),cudaMemcpyHostToDevice,context.stream()));
  for(int offset=0;offset<columns;offset+=tile_width){const int cols=std::min(tile_width,columns-offset);if(tiled)theta_a_action_tile(context,state,d.first_i,d.k_i,parameter,in+size_t(offset)*n,out+size_t(offset)*n,cols);else for(int col=0;col<cols;++col)theta_a_action(context,state,d.first_i,d.k_i,parameter,in+size_t(offset+col)*n,out+size_t(offset+col)*n);}
  Matrix result(n,columns);double in0{},in1{},out0{},out1{};check_dt(cudaMemcpyAsync(result.a.data(),out,count*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check_dt(cudaMemcpyAsync(&in0,in_raw,sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check_dt(cudaMemcpyAsync(&in1,in_raw+count+1,sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check_dt(cudaMemcpyAsync(&out0,out_raw,sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check_dt(cudaMemcpyAsync(&out1,out_raw+count+1,sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check_dt(cudaStreamSynchronize(context.stream()));cudaFree(out_raw);cudaFree(in_raw);if(in0!=guard0||in1!=guard1||out0!=guard0||out1!=guard1)throw std::runtime_error("theta action tile test canary overwritten");return result;
}

std::array<Matrix,10> test_theta_a_second_action_terms(
    GpuContext& context,const WorkerInput& worker,const Matrix& knots,
    const GlobalParameters& parameters,int i,int j,const Matrix& rhs,bool tiled) {
  if((i<0||i>1)||(j<0||j>1)||rhs.rows!=worker.sites.rows) throw std::runtime_error("invalid second action term test input");
  const auto s=build_device_mu_sigma_local(context,worker,knots,parameters);
  const auto d=compute_device_b_pair_derivatives(context,parameters.nu2,parameters.alpha,parameters.range,i,j);
  const int n=s.n,cols=rhs.cols; const size_t count=size_t(n)*cols;
  // Test-only red zones catch an action tile extending past its n-by-columns
  // operands; the production buffers remain unchanged.
  constexpr double guard_lo=0x1.23456789abcp+17,guard_hi=-0x1.987654321fedcp-19;
  double *in_raw{},*out_raw{}; check_dt(cudaMalloc(&in_raw,(count+2)*sizeof(double))); check_dt(cudaMalloc(&out_raw,(count+2)*sizeof(double)));
  double *in=in_raw+1,*out=out_raw+1;
  check_dt(cudaMemcpyAsync(in_raw,&guard_lo,sizeof(double),cudaMemcpyHostToDevice,context.stream()));
  check_dt(cudaMemcpyAsync(in_raw+count+1,&guard_hi,sizeof(double),cudaMemcpyHostToDevice,context.stream()));
  check_dt(cudaMemcpyAsync(out_raw,&guard_lo,sizeof(double),cudaMemcpyHostToDevice,context.stream()));
  check_dt(cudaMemcpyAsync(out_raw+count+1,&guard_hi,sizeof(double),cudaMemcpyHostToDevice,context.stream()));
  check_dt(cudaMemcpyAsync(in,rhs.a.data(),count*sizeof(double),cudaMemcpyHostToDevice,context.stream()));
  std::array<Matrix,10> result; for(auto& x:result)x=Matrix(n,cols);
  const std::array<std::array<const double*,3>,9> terms={{{d.second,context.region_.k,context.region_.b},{d.first_i,d.k_j,context.region_.b},{d.first_i,context.region_.k,d.first_j},{d.first_j,d.k_i,context.region_.b},{context.region_.b,d.k_ij,context.region_.b},{context.region_.b,d.k_i,d.first_j},{d.first_j,context.region_.k,d.first_i},{context.region_.b,d.k_j,d.first_i},{context.region_.b,context.region_.k,d.second}}};
  auto download=[&](int slot,const double* source){check_dt(cudaMemcpyAsync(result[slot].a.data(),source,count*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));};
  if(!tiled) {
    for(int col=0;col<cols;++col) theta_cov_action<<<(n+255)/256,256,0,context.stream()>>>(context.region_.sites,in+size_t(col)*n,n,context.region_.parameter_nu2,context.region_.parameter_alpha_value,context.region_.parameter_range_value,i,j,1,out+size_t(col)*n);
    download(0,out);
    for(int term=0;term<9;++term){ for(int col=0;col<cols;++col){theta_zero_vector<<<(n+255)/256,256,0,context.stream()>>>(out+size_t(col)*n,n);theta_triple_action(context,terms[term][0],terms[term][1],terms[term][2],in+size_t(col)*n,out+size_t(col)*n,-1.);} download(term+1,out); }
  } else {
    // Ensure the dedicated tile storage is allocated without depending on a
    // previous producer phase, then evaluate each algebraic contribution.
    theta_a_second_action_tile(context,s,d,i,j,in,out,cols);
    auto&r=context.region_; check_dt(cudaMemcpyAsync(r.theta_second_tile_in,in,count*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));
    theta_cov_second_action_tile<<<(n*cols+255)/256,256,0,context.stream()>>>(r.sites,r.theta_second_tile_in,n,cols,r.parameter_nu2,r.parameter_alpha_value,r.parameter_range_value,i,j,r.theta_second_tile_out); download(0,r.theta_second_tile_out);
    const double one=1.,minus=-1.,zero=0.;
    for(int term=0;term<9;++term){const auto&t=terms[term];check_dt(cublasDgemm(context.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,s.m,cols,s.n,&one,t[2],s.m,r.theta_second_tile_in,n,&zero,r.theta_second_tile_m0,s.m));check_dt(cublasDgemm(context.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,s.m,cols,s.m,&one,t[1],s.m,r.theta_second_tile_m0,s.m,&zero,r.theta_second_tile_m1,s.m));check_dt(cublasDgemm(context.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,n,cols,s.m,&minus,t[0],s.m,r.theta_second_tile_m1,s.m,&zero,r.theta_second_tile_out,n));download(term+1,r.theta_second_tile_out);}
  }
  double in_lo{},in_hi{},out_lo{},out_hi{};
  check_dt(cudaMemcpyAsync(&in_lo,in_raw,sizeof(double),cudaMemcpyDeviceToHost,context.stream()));
  check_dt(cudaMemcpyAsync(&in_hi,in_raw+count+1,sizeof(double),cudaMemcpyDeviceToHost,context.stream()));
  check_dt(cudaMemcpyAsync(&out_lo,out_raw,sizeof(double),cudaMemcpyDeviceToHost,context.stream()));
  check_dt(cudaMemcpyAsync(&out_hi,out_raw+count+1,sizeof(double),cudaMemcpyDeviceToHost,context.stream()));
  check_dt(cudaStreamSynchronize(context.stream())); check_dt(cudaFree(out_raw)); check_dt(cudaFree(in_raw));
  if(in_lo!=guard_lo||in_hi!=guard_hi||out_lo!=guard_lo||out_hi!=guard_hi)throw std::runtime_error("second action term test canary overwritten");
  return result;
}

}  // namespace fedspatial
