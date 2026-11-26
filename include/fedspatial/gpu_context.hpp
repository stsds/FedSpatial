#pragma once
#include "fedspatial/model.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace fedspatial {
class PProfile {
 public:
  void reset(bool enabled) { enabled_ = enabled; samples_.clear(); }
  bool enabled() const { return enabled_; }
  void add(const char* name, double seconds) {
    if (!enabled_) return;
    auto& sample = samples_[name];
    sample.first += seconds;
    ++sample.second;
  }
  void write_csv(std::ostream& out) const {
    out << "section,calls,wall_seconds\n";
    for (const auto& [name, sample] : samples_)
      out << name << ',' << sample.second << ',' << sample.first << '\n';
  }
 private:
  bool enabled_{};
  std::map<std::string, std::pair<double, long long>> samples_;
};

class PProfileScope {
 public:
  PProfileScope(PProfile& profile, const char* name)
      : profile_(profile), name_(name), start_(std::chrono::steady_clock::now()) {}
  ~PProfileScope() {
    profile_.add(name_, std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count());
  }
 private:
  PProfile& profile_;
  const char* name_;
  std::chrono::steady_clock::time_point start_;
};
// Device-resident gamma sufficient statistics.  The q-vector and q-by-q
// matrix remain owned by the worker RegionGpuState until an explicit
// diagnostic download through compute_gamma_local.
struct DeviceGammaLocalSummary { int q{}; const double* gamma{}; const double* xx{}; };
// The complete FP64 local low-rank state.  This is a non-owning view of the
// RegionGpuState buffers; it is valid until the next builder call on context.
// Unlike LocalComputationResult, creating this view performs no D2H transfer.
struct DeviceMuSigmaLocalState {
  int n{}, m{}, q{};
  const double *B{}, *M{}, *chol_M{}, *minv_B{}, *e{}, *minv_e{};
  const double *bt_minv_e{}, *bt_minv_B{}, *chol_diag{};
  const double *gamma{}, *mu{}, *sigma{}, *parameter_delta{};
  double delta{};
  // [e'M^-1e, log|M|-n log(delta), quadratic, local objective].
  // Production consumers retain this compact checkpoint on device.
  const double* scalar_checkpoint{};
};
// First device-only delta/theta workspace slice.  These are the legacy
// fixed-state definitions r=B mu-e, u=M^-1 r, W=M^-1, and Q=B Sigma B'.
// The backing storage belongs to RegionGpuState and is intentionally exposed
// only through the explicit test downloader below.
struct DeviceDeltaThetaWorkspace {
  int n{}, m{};
  const double *r{}, *u{}, *W{}, *Q{}, *Q_intermediate{};
};
struct DeltaThetaWorkspaceDownload { Matrix r, u, W, Q; };
// Second, still non-reducing, delta/theta device slice.  Parameter order is
// (delta, alpha, range).  B pointers alias the persistent B derivative
// bundle; r/Q storage is owned by RegionGpuState.
struct DeviceDeltaThetaDerivativeWorkspace {
  int n{}, m{};
  std::array<const double*,3> B_i{};
  std::array<std::array<const double*,3>,3> B_ij{};
  std::array<const double*,3> r_i{};
  std::array<std::array<const double*,3>,3> r_ij{};
  std::array<const double*,3> Q_i{};
  std::array<std::array<const double*,3>,3> Q_ij{};
};
struct DeltaThetaDerivativeWorkspaceDownload {
  std::array<Matrix,3> B_i, r_i, Q_i;
  std::array<std::array<Matrix,3>,3> B_ij, r_ij, Q_ij;
};
// Compact, host-visible result of the fully device-resident local P evaluator.
// The 13 doubles are objective, gradient(3), and Hessian(3x3), in that order.
struct DeviceDeltaThetaEvaluation { DeltaThetaLocalResult summary; };
// Non-owning device view used by the synchronous executor.  Unlike the
// compatibility result above, obtaining this view never copies the 13 scalar
// result to the host.
struct DeviceDeltaThetaSummary { const double* values{}; };
// Bounded, deterministic base contraction.  This is intentionally separate
// from the compatibility evaluator while the derivative families are being
// converted: it owns no n-by-n product/result matrix.
struct DeviceDeltaThetaBaseReduction { double quadratic{}, trace_q{}, objective{}; };
// Bounded first-derivative slice that is currently complete for delta.  It
// deliberately has no Q_i or M_i backing matrix: M_delta is consumed through
// M=delta*A+I and W B is reused from the local state.
struct DeviceDeltaThetaDeltaFirstReduction {
  double objective{}, delta_gradient{};
};
// First complete second-order streamed slice.  Its delta/delta entry uses
// only Cholesky solves, n-by-m products, and scalar reductions; it retains no
// W/Q/derivative n-by-n bundle.
struct DeviceDeltaThetaDeltaHessianReduction {
  double objective{}, delta_gradient{}, delta_hessian{};
  // Scalar audit trail for the streamed delta/delta algebra.  These are
  // reductions already computed by the implementation, exposed so parity
  // tests can compare each contraction against a dense reference.
  double trace_wa{}, trace_wawa{}, quadratic_waw{}, trace_wawawq{};
};
// Test-only snapshot taken at the producer's delta/delta boundary, before
// any mixed or theta component reuses the scalar scratch record.
struct DeviceDeltaThetaProducerDeltaDeltaAudit {
  double standalone_delta_hessian{}, device_assembled_delta_hessian{};
  // (base quadratic, base trace-Q, delta score, tr(W), tr(W^2), u'Wu,
  //  tr(W^2 Q), tr(W^3 Q)): exactly the inputs of the device assembler.
  std::array<double, 8> sources{};
};
struct DeviceDeltaThetaProducerBoundaryAudit {
  // Boundaries: base, delta-first, delta-Hessian, final Hdd assembly.
  std::array<std::array<double,11>,4> summary{};
  std::array<std::array<double,13>,4> final{};
};
// Complete streamed delta/theta Hessian slice.  `mixed[i]` is both H(delta,
// theta_i) and H(theta_i,delta), for theta=(alpha, range).  The five audit
// scalars are the five terms in the differentiated delta-gradient identity.
struct DeviceDeltaThetaMixedHessianReduction {
  std::array<double, 2> mixed{};
  std::array<std::array<double, 5>, 2> terms{};
};
// First-order theta slice of the exact fixed-state joint objective.  This is
// deliberately separate from the Hessian summary: it streams every W/A/M/Q
// contraction and retains neither W, Q nor an n-by-n derivative matrix.
struct DeviceDeltaThetaThetaFirstReduction {
  double objective{}, alpha_gradient{}, range_gradient{};
};
// Test-only scalar audit of the streamed theta-first contraction.  Each row
// is (tr(W M_i), 2 r_i' u, u' M_i u, tr(W Q_i), tr(W M_i W Q), gradient).
struct DeviceDeltaThetaThetaFirstDiagnostics {
  std::array<std::array<double, 6>, 2> values{};
};
// (aa, ar, rr), evaluated one pair at a time.  Each row is the exact
// (logdet, residual, covariance-trace) decomposition of that entry.
struct DeviceDeltaThetaThetaHessianReduction {
  std::array<double, 3> hessian{};
  std::array<std::array<double, 3>, 3> terms{};
};
struct RegionGpuState {
  int n{}, m{}, q{};
  const double* host_sites{};
  const double* host_z{};
  const double* host_x{};
  const double* host_knots{};
  std::uint64_t sites_fingerprint{}, z_fingerprint{}, x_fingerprint{}, knots_fingerprint{};
  double* sites{};
  double* z{};
  double* x_col_major{};
  double* knots{};
  // Persistent device-only covariance/B slice workspace.  Matrices use the
  // existing row-major host interpretation; cuBLAS consumes their transpose.
  double* cnk{};
  double* k{};
  double* chol_k{};
  double* b{};
  // Active local P state: Cjj, M, chol(M), and B' M^-1 B.  These replace
  // the former per-call allocations in compute_mu_sigma_local.
  double* cjj{};
  // Persistent FP64 Cjj second derivatives. Host extraction is explicit.
  double *cjj_alpha_alpha{}, *cjj_alpha_range{}, *cjj_range_range{};
  double* m_matrix{};
  double* chol_m{};
  double* bt_rinv_b{};
  // Persistent local mu/sigma builder scratch and state.  Parameter buffers
  // are deliberately resident too: repeated local evaluations do not upload
  // model-sized matrices or allocate temporary device storage.
  double *bk{}, *residual_e{}, *minv_b{}, *minv_e{}, *bt_minv_e{}, *rhs{}, *chol_diag{}, *scalar_checkpoint{};
  double *parameter_gamma{}, *parameter_mu{}, *parameter_sigma{}, *parameter_delta{}, *parameter_theta{};
  // Persistent gamma local evaluator output and M^-1 X scratch.  These are
  // compact at the transport boundary: only gamma_local and gamma_xx move
  // between worker GPUs.
  double *gamma_x_solved{}, *gamma_local{}, *gamma_xx{};
  std::uint64_t parameter_gamma_fingerprint{}, parameter_mu_fingerprint{}, parameter_sigma_fingerprint{};
  double parameter_delta_value{}, parameter_alpha_value{}, parameter_range_value{};
  int parameter_nu2{};
  bool parameters_valid{};
  long long mu_sigma_allocation_count{};
  // A producer-scoped cache only.  It retains the immutable B/M state and a
  // protected copy of B' M^-1 B while the 13-scalar delta/theta record is
  // assembled.  It is never valid across calls, parameters, or workers.
  bool producer_state_cache_active{}, producer_state_cache_valid{};
  double* producer_bt_minv_b{};
  long long producer_state_build_count{}, producer_state_cache_hit_count{},
            producer_b_base_reuse_count{};
  long long producer_scalar_solve_calls{}, producer_tiled_solve_calls{},
            producer_tiled_rhs_columns{};
  // Producer-only host counters.  They are diagnostic only and do not alter
  // the FP64 arithmetic or reduction order.
  long long producer_covariance_action_tiles{}, producer_theta_action_gemm_calls{},
            producer_second_derivative_actions{};
  // Persistent first delta/theta workspace.  This deliberately stops before
  // B_i/B_ij, r_i/Q_i, scalar reductions, or aggregation.
  double *delta_theta_r{}, *delta_theta_u{}, *delta_theta_w{}, *delta_theta_q{}, *delta_theta_q_intermediate{};
  long long delta_theta_workspace_allocation_count{};
  // Optional production solve cache: exactly one red-zoned FP64 W=M^-1.
  // It is separate from the legacy diagnostic W/Q workspace and is selected
  // only by local_solve_strategy=dense_inverse_gemm.
  double *local_dense_w_raw{}, *local_dense_w{}, *local_dense_solve_raw{}, *local_dense_solve{};
  size_t local_dense_w_capacity{}, local_dense_solve_capacity{};
  long long local_dense_w_allocations{}, local_dense_w_builds{}, local_dense_w_reuses{},
            local_dense_w_gemm_calls{}, local_dense_w_avoided_trsm{}, local_dense_w_fallbacks{};
  // chol_m is rebuilt in place; a dense inverse cannot survive that rebuild.
  bool local_dense_w_valid{};
  // Persistent B-derived fixed-state terms for the legacy three-parameter
  // delta/theta equations.  The zero B matrix represents delta derivatives.
  double *delta_theta_b_zero{}, *delta_theta_r_first{}, *delta_theta_r_second{};
  double *delta_theta_q_first{}, *delta_theta_q_second{}, *delta_theta_q_scratch0{}, *delta_theta_q_scratch1{};
  long long delta_theta_derivative_workspace_allocation_count{};
  // The P producer owns two non-aliasing compact records.  `final` is the
  // exact 13-double transport layout; `component_scratch` is diagnostic-only
  // scalar storage and is never gathered.
  double *delta_theta_reduction_scratch{}, *delta_theta_final{},
         *delta_theta_component_scratch{};
  // Transitional aliases retained while the diagnostic component wrappers
  // are being moved onto component_scratch/final.  They are compact only.
  double *delta_theta_summary{}, *delta_theta_theta_first_mm{};
  long long delta_theta_reduction_allocation_count{};
  // Exact element capacities of the compact producer records.  These are
  // retained for allocation-path audits even when a completed pair cache is
  // released before the global-K phase.
  size_t delta_theta_reduction_scratch_capacity{};
  size_t delta_theta_final_capacity{};
  size_t delta_theta_component_scratch_capacity{};
  // ceil(n/tile)^2 FP64 partials for the bounded base contraction.
  double* delta_theta_stream_partials{};
  size_t delta_theta_stream_partials_capacity{};
  // First-order-only B derivative state.  b_first is the sole new n-by-m
  // allocation; bk is borrowed as transient solve scratch after local-state
  // consumers finish.  It never aliases the combined B derivative bundle.
  double *b_first{}, *k_first{};
  long long b_first_derivative_allocation_count{};
  // Bounded pairwise theta workspace.  It is the only derivative residency
  // reachable from the streamed Hessian: B_i, B_j, and B_ij.  The base Cnk
  // and bk buffers are reused as construction scratch.
  double *b_pair_first{}, *b_pair_second{}, *k_pair{}, *k_second{};
  long long b_pair_derivative_allocation_count{};
  size_t b_pair_nm_capacity{}, b_pair_mm_capacity{};
  // Dedicated mixed-Hessian action tile: two n-by-tile matrices (input and
  // output) plus two m-by-tile GEMM intermediates.  Never aliases B/M state.
  double *mixed_action_tile_in{}, *mixed_action_tile_out{}, *mixed_action_tile_m0{}, *mixed_action_tile_m1{};
  size_t mixed_action_tile_n_capacity{}, mixed_action_tile_m_capacity{};
  long long mixed_action_tile_allocation_count{}, mixed_action_tile_calls{},
            mixed_action_tile_rhs_columns{}, mixed_action_trace_tile_calls{},
            mixed_bi_w2b_gemm_calls{}, mixed_v_tz_tile_calls{},
            mixed_v_tz_rhs_columns{};
  // Performance-oriented covariance derivative caches.  The active first
  // derivative and all three symmetric second derivatives remain resident.
  double* theta_cov_first_matrix[2]{}; // alpha, range
  double* theta_cov_second_matrix[3]{}; // (aa, ar, rr)
  size_t theta_cov_first_matrix_capacity{};
  size_t theta_cov_second_matrix_capacity{};
  int theta_cov_first_matrix_nu2{};
  double theta_cov_first_matrix_alpha{}, theta_cov_first_matrix_range{};
  std::uint64_t theta_cov_first_matrix_sites_fingerprint{};
  bool theta_cov_first_matrix_valid[2]{};
  bool theta_cov_second_matrix_valid[3]{};
  int theta_cov_second_matrix_nu2{};
  double theta_cov_second_matrix_alpha{}, theta_cov_second_matrix_range{};
  std::uint64_t theta_cov_second_matrix_sites_fingerprint{};
  long long theta_cov_first_matrix_allocations{}, theta_cov_first_matrix_builds{},
            theta_cov_first_matrix_gemm_calls{};
  long long theta_cov_second_matrix_allocations{}, theta_cov_second_matrix_builds{},
            theta_cov_second_matrix_gemm_calls{};
  // The second-theta-action candidate must never borrow the first-action
  // tiles: log-Hessian input/output can otherwise overlap bk/cnk lifetime.
  // Each raw allocation has a leading and trailing FP64 red-zone; the usable
  // pointer is raw+1.  Capacities count usable elements, not bytes.
  double *theta_second_tile_in_raw{}, *theta_second_tile_out_raw{},
         *theta_second_tile_m0_raw{}, *theta_second_tile_m1_raw{};
  double *theta_second_tile_in{}, *theta_second_tile_out{},
         *theta_second_tile_m0{}, *theta_second_tile_m1{};
  size_t theta_second_tile_n_capacity{}, theta_second_tile_m_capacity{};
  long long theta_second_tile_allocation_count{}, theta_second_tile_calls{},
            theta_second_tile_rhs_columns{}, theta_second_tile_guard_checks{},
            // Three GEMMs per triple-product term.  This is kept separate
            // from Q-panel GEMMs so launch/FLOP accounting cannot double
            // count the two independent theta-Hessian families.
            theta_second_tile_gemm_calls{};
  // Opt-in batched first/middle stages for the nine M_ij triple products.
  // Final GEMMs still accumulate serially into theta_second_tile_out.
  double *theta_second_batched_m0_raw{}, *theta_second_batched_m1_raw{};
  double *theta_second_batched_m0{}, *theta_second_batched_m1{};
  const double **theta_second_batched_a_array{}, **theta_second_batched_b_array{};
  double **theta_second_batched_c_array{};
  size_t theta_second_batched_m_capacity{};
  long long theta_second_batched_allocation_count{}, theta_second_batched_guard_checks{},
            theta_second_batched_logical_gemm_calls{}, theta_second_batched_actual_gemm_calls{};
  long long theta_first_trace_tile_calls{}, theta_first_trace_rhs_columns{},
            theta_first_t1_gemm_calls{}, theta_first_t2_tile_calls{},
            theta_first_t2_rhs_columns{};
  long long theta_hessian_aa_log_tile_calls{}, theta_hessian_aa_log_rhs_columns{},
            theta_hessian_ar_log_tile_calls{}, theta_hessian_ar_log_rhs_columns{},
            theta_hessian_rr_log_tile_calls{}, theta_hessian_rr_log_rhs_columns{};
  long long theta_hessian_q_tile_calls{}, theta_hessian_q_rhs_columns{};
  // Q/W Hessian panel candidate: counts only the new GEMM-backed actions,
  // not the pre-existing matrix-RHS triangular solves.
  long long theta_hessian_q_panel_tile_calls{}, theta_hessian_q_panel_rhs_columns{},
            theta_hessian_q_panel_gemm_calls{};
  // Opt-in Q-base reuse cache.  A single red-zoned n-by-tile panel preserves
  // B Sigma B' W before its three independent residual-Hessian consumers.
  // It is deliberately distinct from bk/cnk and all action scratch.
  double *theta_q_base_cache_raw{}, *theta_q_base_cache{};
  size_t theta_q_base_cache_capacity{};
  long long theta_q_base_cache_allocation_count{}, theta_q_base_cache_guard_checks{},
            theta_q_base_logical_actions{}, theta_q_base_actual_actions{},
            theta_q_base_logical_solves{}, theta_q_base_actual_solves{},
            theta_q_base_cache_copies{};
  // Test-gated Q-second/Q-first super-panel.  Three output regions preserve
  // family-local reductions; eight m-by-tile intermediate regions make the
  // two batched GEMM stages non-overlapping.
  double *theta_q_super_out_raw{}, *theta_q_super_m0_raw{}, *theta_q_super_m1_raw{};
  double *theta_q_super_out{}, *theta_q_super_m0{}, *theta_q_super_m1{};
  const double **theta_q_super_a_array{}, **theta_q_super_b_array{};
  double **theta_q_super_c_array{};
  size_t theta_q_super_out_capacity{}, theta_q_super_m_capacity{};
  long long theta_q_super_allocation_count{}, theta_q_super_guard_checks{},
            theta_q_super_logical_gemm_calls{}, theta_q_super_actual_gemm_calls{},
            theta_q_super_output_copies{};
  // Designated-aggregator, persistent FP64 storage for the global-K P
  // correction: Cholesky/inverse, six ordered product temporaries, and W mu.
  double *global_k_chol{}, *global_k_inverse{}, *global_k_products{}, *global_k_u{};
  long long global_k_correction_allocation_count{};
  // Lazily allocated device-only B derivative bundle and two n-by-m scratch
  // matrices.  They are not allocated by bind_region, avoiding an implicit
  // large-memory commitment for paths that do not evaluate derivatives.
  double *cnk_alpha{}, *cnk_range{}, *cnk_alpha_alpha{}, *cnk_alpha_range{}, *cnk_range_range{};
  double *k_alpha{}, *k_range{}, *k_alpha_alpha{}, *k_alpha_range{}, *k_range_range{};
  double *b_alpha{}, *b_range{}, *b_alpha_alpha{}, *b_alpha_range{}, *b_range_alpha{}, *b_range_range{};
  double *b_scratch0{}, *b_scratch1{};
  // Persistent A derivative outputs and one n-by-n accumulation scratch.
  // They are populated entirely from the B/K/Cjj device bundles.
  double *a_value{}, *a_alpha{}, *a_range{}, *a_alpha_alpha{}, *a_alpha_range{}, *a_range_alpha{}, *a_range_range{}, *a_scratch{};
  // Persistent M derivative bundle. Mixed delta derivatives alias the
  // corresponding first-derivative storage to preserve exact symmetry.
  double *m_delta{}, *m_alpha{}, *m_range{}, *m_dd{}, *m_da{}, *m_dr{}, *m_aa{}, *m_ar{}, *m_ra{}, *m_rr{};
  double* potrf_work{};
  int* potrf_info{};
  size_t potrf_work_capacity{};
  long long raw_h2d_bytes{};
  long long parameter_h2d_bytes{};
  // Device-to-host values used by active implementation internals rather
  // than an explicit diagnostic extractor (for example Cholesky diagonals).
  long long host_compute_d2h_bytes{};
  // Full device-result downloads are deliberately opt-in.  The active
  // device B slice never uses this path; it is exposed for parity tests.
  long long result_d2h_bytes{};
  long long cjj_second_allocation_count{};
  long long b_derivative_allocation_count{};
  long long a_derivative_allocation_count{};
  long long m_derivative_allocation_count{};
  bool initialized{};
  // Reduced-precision scratch is opt-in. The canonical state and all
  // factorization/solve buffers remain FP64 so promotion is lossless.
  void *precision_a{}, *precision_b{}, *precision_c{};
  size_t precision_a_bytes{}, precision_b_bytes{}, precision_c_bytes{};
  long long precision_gemm_calls{}, precision_promotions{};
  // These three super-panel allocations are intentionally not in the legacy
  // GpuContext free-list.  They own their raw red-zone bases and are released
  // when the region dies (after its owning context selected the device).
  ~RegionGpuState() {
    if (local_dense_w_raw) cudaFree(local_dense_w_raw);
    if (local_dense_solve_raw) cudaFree(local_dense_solve_raw);
    if (theta_second_batched_m0_raw) cudaFree(theta_second_batched_m0_raw);
    if (theta_second_batched_m1_raw) cudaFree(theta_second_batched_m1_raw);
    if (theta_second_batched_a_array) cudaFree(const_cast<double**>(theta_second_batched_a_array));
    if (theta_second_batched_b_array) cudaFree(const_cast<double**>(theta_second_batched_b_array));
    if (theta_second_batched_c_array) cudaFree(theta_second_batched_c_array);
    if (theta_q_super_out_raw) cudaFree(theta_q_super_out_raw);
    if (theta_q_super_m0_raw) cudaFree(theta_q_super_m0_raw);
    if (theta_q_super_m1_raw) cudaFree(theta_q_super_m1_raw);
    if (theta_q_super_a_array) cudaFree(const_cast<double**>(theta_q_super_a_array));
    if (theta_q_super_b_array) cudaFree(const_cast<double**>(theta_q_super_b_array));
    if (theta_q_super_c_array) cudaFree(theta_q_super_c_array);
    for (double* p : theta_cov_first_matrix) if (p) cudaFree(p);
    for (double* p : theta_cov_second_matrix) if (p) cudaFree(p);
  }
};
struct DeviceBSlice { int n{}, m{}; const double* cnk{}; const double* k{}; const double* chol_k{}; const double* b{}; };
struct DeviceBFirstDerivativeSlice { int n{}, m{}; const double* derivative{}; };
struct DeviceBPairDerivativeSlice {
  int n{}, m{};
  const double *first_i{}, *first_j{}, *second{};
  const double *k_i{}, *k_j{}, *k_ij{};
};
// Device-only B derivative bundle, parameter order (alpha, range).  It is
// intentionally not a host BSecondDerivatives alias: consumers can retain
// these FP64 matrices on the owning worker device.
struct DeviceBDerivativeSlice { int n{}, m{}; const double *alpha{}, *range{}, *alpha_alpha{}, *alpha_range{}, *range_alpha{}, *range_range{}; };
struct DeviceCjjSecondDerivativeSlice { int n{}; const double *alpha_alpha{}, *alpha_range{}, *range_alpha{}, *range_range{}; };
// Device-only A derivatives, in (alpha, range) order.  The first and second
// derivative matrices remain GPU-resident until an explicit diagnostic download.
struct DeviceADerivativeSlice { int n{}; const double *value{}, *alpha{}, *range{}, *alpha_alpha{}, *alpha_range{}, *range_alpha{}, *range_range{}; };
struct DeviceMDerivativeSlice { int n{}; const double *delta{}, *alpha{}, *range{}, *dd{}, *da{}, *dr{}, *ad{}, *aa{}, *ar{}, *rd{}, *ra{}, *rr{}; };

class GpuContext {
 public:
  explicit GpuContext(int device_id);
  ~GpuContext();
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;
  int device_id() const;
  cudaStream_t stream() const;
  cublasHandle_t cublas() const;
  cusolverDnHandle_t cusolver() const;
  PProfile& profile();
  // Default is the established streamed path.  Dense W requires explicit
  // caller opt-in through this typed policy boundary.
  void set_local_solve_configuration(LocalSolveConfiguration configuration);
  LocalSolveConfiguration local_solve_configuration() const;
  void set_precision_policy(PrecisionPolicy policy);
  PrecisionPolicy precision_policy() const;
  void bind_region(const WorkerInput& worker, const Matrix& knots);
  const RegionGpuState& region() const;
  long long raw_h2d_bytes() const;
  long long parameter_h2d_bytes() const;
  long long host_compute_d2h_bytes() const;
  long long result_d2h_bytes() const;
  long long cjj_second_allocation_count() const;
  long long b_derivative_allocation_count() const;
  long long b_first_derivative_allocation_count() const;
  long long b_pair_derivative_allocation_count() const;
  long long a_derivative_allocation_count() const;
  long long m_derivative_allocation_count() const;
  long long mu_sigma_allocation_count() const;
  long long delta_theta_workspace_allocation_count() const;
  long long delta_theta_derivative_workspace_allocation_count() const;
  long long delta_theta_reduction_allocation_count() const;
  long long global_k_correction_allocation_count() const;
  void record_model_d2h_bytes(size_t bytes);
  long long right_solve_allocation_count() const;
  long long right_solve_factor_upload_count() const;
 private:
  friend Matrix gpu_right_spd_solve(GpuContext&, const Matrix&, const Matrix&);
  friend DeviceBSlice compute_device_b_slice(GpuContext&, int, double, double);
  friend Matrix download_device_b_slice(GpuContext&, const DeviceBSlice&);
  friend DeviceBDerivativeSlice compute_device_b_derivatives(GpuContext&, int, double, double);
  friend DeviceBFirstDerivativeSlice compute_device_b_first_derivative(GpuContext&, int, double, double, int);
  friend DeviceBPairDerivativeSlice compute_device_b_pair_derivatives(GpuContext&, int, double, double, int, int);
  friend std::array<Matrix,10> test_theta_a_second_action_terms(
      GpuContext&, const WorkerInput&, const Matrix&, const GlobalParameters&, int, int,
      const Matrix&, bool);
  friend void release_device_delta_theta_pair_workspace(GpuContext&);
  friend BSecondDerivatives download_device_b_derivatives(GpuContext&, const DeviceBDerivativeSlice&);
  friend DeviceCjjSecondDerivativeSlice compute_device_cjj_second_derivatives(GpuContext&, int, double, double);
  friend CjjSecondDerivatives download_device_cjj_second_derivatives(GpuContext&, const DeviceCjjSecondDerivativeSlice&);
  friend DeviceADerivativeSlice compute_device_a_derivatives(GpuContext&, int, double, double);
  friend ASecondDerivatives download_device_a_second_derivatives(GpuContext&, const DeviceADerivativeSlice&);
  friend std::pair<AFirstDerivatives, ASecondDerivatives> download_device_a_derivatives(GpuContext&, const DeviceADerivativeSlice&);
  friend DeviceMDerivativeSlice compute_device_m_derivatives(GpuContext&, int, double, double, double);
  friend DeviceDeltaThetaEvaluation compute_device_delta_theta_local(GpuContext&, const WorkerInput&, const Matrix&, const GlobalParameters&);
  friend DeviceDeltaThetaSummary compute_device_delta_theta_summary(GpuContext&, const WorkerInput&, const Matrix&, const GlobalParameters&);
  friend DeviceDeltaThetaBaseReduction compute_device_delta_theta_base_reduction_streamed(GpuContext&, const WorkerInput&, const Matrix&, const GlobalParameters&);
  friend DeviceDeltaThetaDeltaFirstReduction compute_device_delta_theta_delta_first_reduction_streamed(GpuContext&, const WorkerInput&, const Matrix&, const GlobalParameters&);
  friend DeviceDeltaThetaDeltaHessianReduction compute_device_delta_theta_delta_hessian_reduction_streamed(GpuContext&, const WorkerInput&, const Matrix&, const GlobalParameters&);
  friend DeviceDeltaThetaProducerDeltaDeltaAudit compute_device_delta_theta_producer_delta_delta_audit_for_test(GpuContext&, const WorkerInput&, const Matrix&, const GlobalParameters&);
  friend DeviceDeltaThetaMixedHessianReduction compute_device_delta_theta_mixed_hessian_reduction_streamed(GpuContext&, const WorkerInput&, const Matrix&, const GlobalParameters&);
  friend DeviceDeltaThetaThetaFirstReduction compute_device_delta_theta_theta_first_reduction_streamed(GpuContext&, const WorkerInput&, const Matrix&, const GlobalParameters&);
  friend DeviceDeltaThetaThetaFirstDiagnostics compute_device_delta_theta_theta_first_diagnostics_for_test(GpuContext&, const WorkerInput&, const Matrix&, const GlobalParameters&);
  friend DeviceDeltaThetaThetaHessianReduction compute_device_delta_theta_theta_hessian_reduction_streamed(GpuContext&, const WorkerInput&, const Matrix&, const GlobalParameters&);
  friend void apply_device_global_k_correction(GpuContext&, double*);
  friend DeviceGammaLocalSummary compute_device_gamma_local(GpuContext&, const WorkerInput&, const Matrix&, const GlobalParameters&);
  friend std::pair<MFirstDerivatives, MSecondDerivatives> download_device_m_derivatives(GpuContext&, const DeviceMDerivativeSlice&);
  int device_id_;
  cudaStream_t stream_{};
  cublasHandle_t cublas_{};
  cusolverDnHandle_t cusolver_{};
  PProfile profile_{};
  LocalSolveConfiguration local_solve_configuration_{};
  PrecisionPolicy precision_policy_{};
  double* right_solve_factor_{};
  double* right_solve_rhs_{};
  size_t right_solve_factor_capacity_{};
  size_t right_solve_rhs_capacity_{};
  std::vector<double> right_solve_factor_host_;
  std::vector<double> right_solve_rhs_host_;
  long long right_solve_allocation_count_{};
  long long right_solve_factor_upload_count_{};
  RegionGpuState region_{};
};

// Executes a beta=0 GEMM through the configured reduced-precision policy and
// converts the FP32 accumulator back to the canonical FP64 device state.
// Returns false in FP64 mode so callers can use their established Dgemm path.
bool gpu_precision_gemm(GpuContext&, cublasOperation_t, cublasOperation_t,
                        int rows, int columns, int inner,
                        const double* a, int lda, const double* b, int ldb,
                        double* c, int ldc);

// One isolated stream/handle set is owned for each logical worker.  Device IDs
// are assigned in vector/worker-ID order and may repeat for same-GPU streams.
// Each local stage runs concurrently; all host reductions are deterministic.
class NWorkerPSynchronousExecutor {
 public:
  explicit NWorkerPSynchronousExecutor(std::vector<int> device_ids);
  ~NWorkerPSynchronousExecutor();
  PSynchronousRoundResult run_round(const std::vector<WorkerInput>& workers,
                                    const Matrix& knots,
                                    const GlobalParameters& parameters);
  PWorkerRoundCheckpoints run_round_with_checkpoints(
      const std::vector<WorkerInput>& workers, const Matrix& knots,
      const GlobalParameters& parameters);
  int worker_count() const;
  // Transport accounting deliberately excludes initialization uploads.  These
  // counters cover compact sufficient statistics only.
  long long peer_summary_bytes() const { return peer_summary_bytes_; }
  long long pinned_summary_bytes() const { return pinned_summary_bytes_; }
 public:  // executor transport internals; not part of the stable facade
  std::vector<std::unique_ptr<GpuContext>> contexts_;
  int round_sequence_{};
  // Owned by worker 0's GPU.  It is reused across rounds and never holds a
  // raw region or a model-sized local matrix.
  double* compact_aggregation_{};
  size_t compact_aggregation_capacity_{};
  // Central-only FP64 scratch for the K correction and 3x3 Newton update.
  // It contains no raw worker region data.
  double* global_update_{};
  size_t global_update_capacity_{};
  void* pinned_compact_{};
  size_t pinned_compact_capacity_{};
  long long peer_summary_bytes_{};
  long long pinned_summary_bytes_{};
};

// Backward-compatible two-worker facade.
class TwoWorkerPSynchronousExecutor {
 public:
  TwoWorkerPSynchronousExecutor(int device_id_worker0, int device_id_worker1);
  PSynchronousRoundResult run_round(const std::array<WorkerInput, 2>& workers,
                                    const Matrix& knots,
                                    const GlobalParameters& parameters);
  PTwoWorkerRoundCheckpoints run_round_with_checkpoints(
      const std::array<WorkerInput, 2>& workers, const Matrix& knots,
      const GlobalParameters& parameters);
 private:
  std::unique_ptr<NWorkerPSynchronousExecutor> executor_;
};
// Stable host reduction used by synchronous worker orchestration. Inputs are
// accumulated in vector index (worker-ID) order.
Matrix deterministic_worker_order_mean(const std::vector<Matrix>& values);

// The P-residual local calculation and its parity diagnostics share one GPU path.
LocalComputationResult compute_mu_sigma_local(GpuContext& context,
                                              const WorkerInput& worker,
                                              const Matrix& knots,
                                              const GlobalParameters& parameters);
// Test-only frozen pre-persistent oracle; preserves the historical public
// host layout while the active implementation remains device-resident.
LocalComputationResult compute_mu_sigma_local_legacy_oracle(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters);
// Active device builder.  It retains every low-rank local component on the
// worker GPU and intentionally performs no result download.
DeviceMuSigmaLocalState build_device_mu_sigma_local(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters);
// Explicit compatibility/oracle extraction.  Production device consumers
// should retain DeviceMuSigmaLocalState instead of calling this function.
LocalComputationResult download_device_mu_sigma_local(
    GpuContext& context, const DeviceMuSigmaLocalState& state);
// Allocation-light regression of the production compact reducers at m=2000.
double compact_reduction_m2000_max_abs_error(GpuContext& context);
// Builds only the first persistent GPU delta/theta workspace slice.  It
// consumes the existing device mu/sigma state and performs no D2H transfer.
DeviceDeltaThetaWorkspace build_device_delta_theta_workspace(
    GpuContext& context, const DeviceMuSigmaLocalState& state);
// Test-only audit hook for the separately-owned production dense-W cache.
// It has no effect on the default streamed solve selection.
void build_device_local_dense_w_for_test(
    GpuContext& context, const DeviceMuSigmaLocalState& state);
// Test-only diagnostic extraction; production code must retain the device view.
DeltaThetaWorkspaceDownload download_device_delta_theta_workspace_for_test(
    GpuContext& context, const DeviceDeltaThetaWorkspace& workspace);
// Builds B_i/B_ij, r_i/r_ij, and Q_i/Q_ij only.  It deliberately excludes
// trace/quadratic reductions and all host aggregation.
DeviceDeltaThetaDerivativeWorkspace build_device_delta_theta_derivative_workspace(
    GpuContext& context, const DeviceMuSigmaLocalState& state);
DeltaThetaDerivativeWorkspaceDownload
download_device_delta_theta_derivative_workspace_for_test(
    GpuContext& context, const DeviceDeltaThetaDerivativeWorkspace& workspace);
// Active local P evaluator. Its only host transfer is the compact 13-double
// objective/gradient/Hessian summary; all fixed-state and derivative matrices
// stay in RegionGpuState.
DeviceDeltaThetaEvaluation compute_device_delta_theta_local(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters);
DeviceDeltaThetaSummary compute_device_delta_theta_summary(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters);
// Computes r'M^-1r and tr(M^-1 Q) in fixed 128x128 tiles.  The only new
// storage is ceil(n/128)^2 FP64 partials, reused on every call.
DeviceDeltaThetaBaseReduction compute_device_delta_theta_base_reduction_streamed(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters);
// Fixed-order FP64 delta first derivative paired with the streamed base
// contraction.  It allocates no derivative n-by-n matrix or Q_i bundle.
DeviceDeltaThetaDeltaFirstReduction
compute_device_delta_theta_delta_first_reduction_streamed(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters);
DeviceDeltaThetaDeltaHessianReduction
compute_device_delta_theta_delta_hessian_reduction_streamed(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters);
DeviceDeltaThetaProducerDeltaDeltaAudit
compute_device_delta_theta_producer_delta_delta_audit_for_test(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters);
DeviceDeltaThetaProducerBoundaryAudit
compute_device_delta_theta_producer_boundary_audit_for_test(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters);
DeviceDeltaThetaMixedHessianReduction
compute_device_delta_theta_mixed_hessian_reduction_streamed(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters);
DeviceDeltaThetaThetaFirstReduction
compute_device_delta_theta_theta_first_reduction_streamed(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters);
DeviceDeltaThetaThetaFirstDiagnostics
compute_device_delta_theta_theta_first_diagnostics_for_test(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters);
DeviceDeltaThetaThetaHessianReduction
compute_device_delta_theta_theta_hessian_reduction_streamed(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters);
// Adds the legacy global-K gradient/Hessian correction to a 13-double
// aggregated local-P summary on this context's GPU.  The context must already
// own the matching device parameters, knots, K, and K derivative bundle.
void apply_device_global_k_correction(GpuContext& context, double* aggregated_summary);
// Frozen host global-K equation, exposed only to the focused correction
// parity harness.  It consumes/returns compact three-parameter quantities.
void compute_legacy_global_k_correction_for_test(
    GpuContext& context, const Matrix& knots, const GlobalParameters& parameters,
    Matrix* gradient, Matrix* hessian);
// Frozen host equation implementation, used solely by local evaluator parity.
DeltaThetaLocalResult compute_delta_theta_local_legacy_oracle(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters, const PGlobalDerivativeWorkspace& global_workspace);
GammaLocalSummary compute_gamma_local(GpuContext& context,
                                      const WorkerInput& worker,
                                      const Matrix& knots,
                                      const GlobalParameters& parameters);
// Active gamma evaluator: all model-sized state and its compact output remain
// on the owning GPU.  This is the worker-side input to GPU aggregation.
DeviceGammaLocalSummary compute_device_gamma_local(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters);
GammaLocalSummary compute_ind_gamma_local(GpuContext& context, const WorkerInput& worker,
                                           const GlobalParameters& parameters);
IndDeltaThetaLocalResult compute_ind_delta_theta_local(GpuContext& context,
                                                        const WorkerInput& worker,
                                                        const GlobalParameters& parameters);
PSynchronousRoundResult run_ind_synchronous_round(
    GpuContext& context, const std::vector<WorkerInput>& workers,
    const GlobalParameters& parameters);
DeltaThetaLocalResult compute_delta_theta_local(GpuContext& context,
                                                const WorkerInput& worker,
                                                const Matrix& knots,
                                                const GlobalParameters& parameters,
                                                const PGlobalDerivativeWorkspace& global_workspace);
// Compatibility entry point for existing local delta/theta parity tests.
DeltaThetaLocalResult compute_delta_theta_local(GpuContext& context,
                                                const WorkerInput& worker,
                                                const Matrix& knots,
                                                const GlobalParameters& parameters);
PSynchronousRoundResult run_p_synchronous_round(
    GpuContext& context, const std::vector<WorkerInput>& workers,
    const Matrix& knots, const GlobalParameters& parameters);
BSecondDerivatives compute_b_second_derivatives(GpuContext& context,
                                                const WorkerInput& worker,
                                                const Matrix& knots,
                                                const GlobalParameters& parameters,
                                                const PGlobalDerivativeWorkspace& global_workspace,
                                                const CnkDerivativeWorkspace& cnk_workspace);
BSecondDerivatives compute_b_second_derivatives(GpuContext& context,
                                                const WorkerInput& worker,
                                                const Matrix& knots,
                                                const GlobalParameters& parameters,
                                                const PGlobalDerivativeWorkspace& global_workspace);
// Compatibility entry point for existing B-Hessian parity fixtures.
BSecondDerivatives compute_b_second_derivatives(GpuContext& context,
                                                const WorkerInput& worker,
                                                const Matrix& knots,
                                                const GlobalParameters& parameters);
PGlobalDerivativeWorkspace make_p_global_derivative_workspace(
    GpuContext& context, const Matrix& knots, const GlobalParameters& parameters);
CnkDerivativeWorkspace make_cnk_derivative_workspace(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters);
CnkDerivativeWorkspace compute_cnk_derivatives_fused(
    GpuContext& context, const Matrix& sites, const Matrix& knots, int nu2,
    double alpha, double range);
Matrix gpu_right_spd_solve(GpuContext& context, const Matrix& x,
                           const Matrix& chol_k);
// Device-only covariance Cnk, K factorization, and B=Cnk K^-1 slice.  It
// consumes RegionGpuState raw inputs and performs no host matrix transfer.
DeviceBSlice compute_device_b_slice(GpuContext& context, int nu2,
                                    double alpha, double range);
// Explicit diagnostic extraction only; production paths must not call this.
Matrix download_device_b_slice(GpuContext& context, const DeviceBSlice& slice);
DeviceBDerivativeSlice compute_device_b_derivatives(GpuContext& context, int nu2,
                                                     double alpha, double range);
// Builds dB/dalpha (parameter=0) or dB/drange (parameter=1) in one reusable
// FP64 n-by-m buffer.  It never calls or allocates the combined B bundle.
DeviceBFirstDerivativeSlice compute_device_b_first_derivative(
    GpuContext& context, int nu2, double alpha, double range, int parameter);
// Builds only one ordered derivative pair (B_i, B_j, B_ij), reusing three
// n-by-m buffers.  It never calls or allocates the legacy 13-buffer bundle.
DeviceBPairDerivativeSlice compute_device_b_pair_derivatives(
    GpuContext& context, int nu2, double alpha, double range, int i, int j);
void release_device_delta_theta_pair_workspace(GpuContext& context);
// Explicit test/oracle extraction only; active device consumers retain the
// DeviceBDerivativeSlice without this D2H operation.
BSecondDerivatives download_device_b_derivatives(GpuContext& context,
                                                  const DeviceBDerivativeSlice& slice);
// Device-only Cjj derivatives in (alpha, range) order. The region must be bound.
DeviceCjjSecondDerivativeSlice compute_device_cjj_second_derivatives(
    GpuContext& context, int nu2, double alpha, double range);
// Explicit test/legacy extraction only; device consumers retain the slice.
CjjSecondDerivatives download_device_cjj_second_derivatives(
    GpuContext& context, const DeviceCjjSecondDerivativeSlice& slice);
CjjSecondDerivatives compute_cjj_second_derivatives(GpuContext& context,
                                                     const WorkerInput& worker,
                                                     const GlobalParameters& parameters);
DeviceADerivativeSlice compute_device_a_derivatives(GpuContext& context, int nu2,
                                                     double alpha, double range);
// Explicit test/oracle extraction only; this is not part of the active A path.
ASecondDerivatives download_device_a_second_derivatives(
    GpuContext& context, const DeviceADerivativeSlice& slice);
std::pair<AFirstDerivatives, ASecondDerivatives> download_device_a_derivatives(
    GpuContext& context, const DeviceADerivativeSlice& slice);
DeviceMDerivativeSlice compute_device_m_derivatives(GpuContext& context, int nu2,
                                                     double delta, double alpha, double range);
std::pair<MFirstDerivatives, MSecondDerivatives> download_device_m_derivatives(
    GpuContext& context, const DeviceMDerivativeSlice& slice);
ASecondDerivatives compute_a_second_derivatives(GpuContext& context,
                                                const WorkerInput& worker,
                                                const Matrix& knots,
                                                const GlobalParameters& parameters);
ASecondDerivatives compute_a_second_derivatives_from_b(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters, const BSecondDerivatives& b_second);
MFirstDerivatives compute_m_first_derivatives(GpuContext& context,
                                              const WorkerInput& worker,
                                              const Matrix& knots,
                                              const GlobalParameters& parameters);
MSecondDerivatives compute_m_second_derivatives(GpuContext& context,
                                                const WorkerInput& worker,
                                                const Matrix& knots,
                                                 const GlobalParameters& parameters);
MSecondDerivatives assemble_m_second_derivatives(
    GpuContext& context, const MFirstDerivatives& first,
    const ASecondDerivatives& second, double delta);
MaternDerivatives matern_with_derivatives(GpuContext& context, const Matrix& x,
                                          const Matrix& y, int nu2,
                                          double alpha, double range);
// Test-only oracle for the mixed-Hessian action primitive; never called by
// the production producer.
Matrix test_theta_a_action_tile(GpuContext& context, const WorkerInput& worker,
                                const Matrix& knots, const GlobalParameters& parameters,
                                int parameter, const Matrix& rhs, int tile_width,
                                bool tiled);
// Test-only decomposition of A_ij*x: item zero is C_ij*x and items 1..9
// are the nine signed low-rank triple-product contributions.
std::array<Matrix,10> test_theta_a_second_action_terms(
    GpuContext& context, const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters, int i, int j, const Matrix& rhs,
    bool tiled);
}
