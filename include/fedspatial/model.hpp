#pragma once
#include "fedspatial/sync_cpu.hpp"
#include <array>
#include <cstddef>
namespace fedspatial {
struct WorkerInput { Matrix sites, z, x; };
// nu2 is twice the Matérn smoothness: 1, 3, and 5 are the Figure 1 cases.
struct GlobalParameters { Matrix gamma, mu, sigma; double delta, alpha, range; int nu2{3}; };
// Public policy boundary for the local M solve.  Launchers may map their
// user-facing option to this enum without exposing producer implementation
// gates.  auto_select is intentionally a streamed-TRSM alias: dense W is
// correctness-only and may be requested solely as dense_inverse_gemm.
enum class LocalSolveStrategy { auto_select, dense_inverse_gemm, streamed_trsm };
struct LocalSolveConfiguration { LocalSolveStrategy strategy{LocalSolveStrategy::dense_inverse_gemm}; size_t dense_memory_budget_bytes{8ull<<30}; };
enum class ArithmeticPrecision { fp64, fp32, mixed_fp16_fp32 };
struct PrecisionPolicy {
  ArithmeticPrecision mode{ArithmeticPrecision::fp64};
  double residual_tolerance{1e-8};
  int iterative_refinement_steps{2};
  bool promote_on_failure{true};
};
struct MuSigmaLocalSummary { Matrix mu, sigma; };
struct GammaLocalSummary { Matrix gamma, xx; };
struct IndDeltaThetaLocalResult { double objective{}; std::array<double,3> gradient{}; std::array<double,9> hessian{}; };
struct DeltaThetaLocalResult { double objective{}; std::array<double, 3> gradient{}; std::array<double, 9> hessian{}; };
struct PSynchronousRoundResult {
  GlobalParameters parameters;
  double accepted_step{};
  int halving_count{};
};
struct PSynchronousRoundTiming {
  double mu_sigma_local_seconds{};
  double gamma_local_seconds{};
  double delta_theta_local_seconds{};
  double aggregation_seconds{};
  double global_update_seconds{};
  double broadcast_seconds{};
  double total_seconds{};
  // Transfer instrumentation for the active N-worker round.  Raw bytes are
  // immutable region uploads; model bytes are parameter/checkpoint traffic.
  long long raw_h2d_bytes{};
  long long model_h2d_bytes{};
  long long model_d2h_bytes{};
  long long host_compute_d2h_bytes{};
  long long peer_summary_bytes{};
  long long pinned_summary_bytes{};
  bool raw_data_stationary{true};
  bool model_data_stationary{true};
};
struct PWorkerRoundCheckpoints {
  GlobalParameters after_mu_sigma;
  GlobalParameters after_gamma;
  PSynchronousRoundResult after_delta_theta;
  PSynchronousRoundTiming timing;
};
using PTwoWorkerRoundCheckpoints = PWorkerRoundCheckpoints;
struct BSecondDerivatives { Matrix kaa, kar, krr, caa, car, crr, baa, bar, bra, brr; };
struct CjjSecondDerivatives { Matrix aa, ar, ra, rr; };
struct ASecondDerivatives { Matrix aa, ar, ra, rr; };
struct AFirstDerivatives { Matrix alpha, range; };
struct MFirstDerivatives { Matrix delta, alpha, range; };
struct MSecondDerivatives { Matrix dd, da, dr, ad, aa, ar, rd, ra, rr; };
// Exact fixed-state derivative data owned by one delta/theta worker evaluation.
// It makes dependencies explicit and prevents lower derivative layers from
// rebuilding B/M state while assembling the same Hessian.
struct PDerivativeWorkspace {
  BSecondDerivatives b_second;
  MFirstDerivatives m_first;
  ASecondDerivatives a_second;
  MSecondDerivatives m_second;
};
struct PGlobalDerivativeWorkspace {
  Matrix k, k_alpha, k_range, k_alpha_alpha, k_alpha_range, k_range_range, chol_k;
};
struct CnkDerivativeWorkspace { Matrix alpha, range, alpha_alpha, alpha_range, range_range; };
// Reusable Figure 1 Matérn primitives. nu2 is twice the smoothness (1,3,5).
struct MaternDerivatives { Matrix value, alpha, range, aa, ar, ra, rr; };
struct LocalComputationResult {
  MuSigmaLocalSummary summary;
  Matrix cstar, B, M, chol_M, e, rinv_B, bt_rinv_B, rinv_e, bt_rinv_e;
  double et_rinv_e{}, logdet_R{}, local_objective{};
};
}
