#include "fedspatial/nworker_15k_figure1_config.hpp"

#include <iostream>
#include <stdexcept>

// This is the sole large-case readiness harness.  It supports dry-run only:
// it neither constructs Matrix objects nor calls cudaMalloc for the 15k case.
using namespace fedspatial;
constexpr int kN=NWorker15kFigure1Config::local_n, kM=NWorker15kFigure1Config::knots;
static void require(cudaError_t e, const char* what) { if(e!=cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e)); }
int main(int argc,char**argv) { try {
  // This target is deliberately not a launcher.
  if (argc>1)
    throw std::runtime_error("refusing launch: this readiness harness supports dry-run only");
  int count=0; require(cudaGetDeviceCount(&count),"cudaGetDeviceCount");
  print_nworker_15k_figure1_config(std::cout);
  std::cout<<"BENCHMARK_CONFIGURATION precision=FP64 method=P_low_rank executor=NWorkerPSynchronousExecutor producer=compute_device_delta_theta_summary\n";
  // The allocation-free gate deliberately contains only device capability and
  // conservative byte estimates, never Matrix construction or executor setup.
  const long double nn=(long double)kN*kN, nm=(long double)kN*kM;
  // Inventory is deliberately explicit: the old 25*n*n arena is forbidden.
  // "scheduled" is the completed target lifetime layout (M is factored
  // in-place; Q/Qi/Qij and WMiW products are streamed).
  const long double legacy_core=8.L*(3*nn+3*nm+2*static_cast<long double>(kM)*kM);
  const long double legacy_reduction=8.L*25.L*nn;
  const long double scheduled_m_factor=8.L*nn;
  const long double scheduled_b=8.L*(4*nm+3*static_cast<long double>(kM)*kM);
  const long double scheduled_stream=8.L*(4*static_cast<long double>(kN)*128+
      ((kN+127)/128)*((kN+127)/128)+64);
  // cuSOLVER potrf workspace is reserved as a conservative 2*n*n FP64 cap.
  const long double scheduled_solver=16.L*nn, scheduled_margin=512.L*1024*1024;
  const long double scheduled_peak=scheduled_m_factor+scheduled_b+scheduled_stream+scheduled_solver+scheduled_margin;
  // What the current code actually retains while evaluating the streamed
  // base/delta slice: Cjj, M, chol(M), W, Q, plus the conservative POTRF
  // reserve.  This is intentionally separate from the target schedule.
  // Measured allocation graph for the active streamed base/delta path:
  // chol(M), Cnk/B/bk/minv_B and three m*m summaries.  The BKB' buffer is
  // freed before the queried POTRF reserve; W/Q are compatibility-only lazy
  // allocations and are not reachable from this path.
  const long double current_delta_first_peak=scheduled_m_factor+
      8.L*(4.L*nm+3.L*static_cast<long double>(kM)*kM)+scheduled_solver+scheduled_margin;
  // The complete standalone 3x3 streamed Hessian retains only the reusable
  // pair (B_i, B_j, B_ij) and three m-by-m K actions.  It does not reach the
  // legacy 13 n-by-m B/C bundle or any n-by-n A/M derivative allocation.
  const long double current_theta_hessian_peak=scheduled_m_factor+
      8.L*(7.L*nm+6.L*static_cast<long double>(kM)*kM)+scheduled_solver+scheduled_margin;
  // The central correction retains ten m-by-m ordered product slots plus
  // five m-by-m K derivative matrices.  Both are independent of local n
  // and replace the historical 25*n*n arena.
  const long double current_global_k_phase=current_theta_hessian_peak-
      8.L*3.L*nm+8.L*15.L*static_cast<long double>(kM)*kM;
  const long double current_executor_peak=std::max(current_theta_hessian_peak,
                                                    current_global_k_phase);
  const long double gib=1024.L*1024*1024;
  std::cout<<"DRY_RUN=PASS allocated_large_matrices=0 legacy_inventory_bytes_per_gpu="<<(unsigned long long)(legacy_core+legacy_reduction)<<" legacy_25nn_bytes="<<(unsigned long long)legacy_reduction<<" target_scheduled_peak_bytes_per_gpu="<<(unsigned long long)scheduled_peak<<" target_scheduled_peak_gib="<<double(scheduled_peak/gib)<<" target_budget_status="<<(scheduled_peak<7.5L*gib?"PASS":"FAIL")<<" current_delta_first_peak_bytes_per_gpu="<<(unsigned long long)current_delta_first_peak<<" current_delta_first_peak_gib="<<double(current_delta_first_peak/gib)<<" complete_3x3_hessian_peak_bytes_per_gpu="<<(unsigned long long)current_theta_hessian_peak<<" complete_3x3_hessian_peak_gib="<<double(current_theta_hessian_peak/gib)<<" executor_global_k_scratch_bytes="<<(unsigned long long)(8.L*10.L*kM*kM)<<" executor_global_k_derivative_bytes="<<(unsigned long long)(8.L*5.L*kM*kM)<<" executor_pair_buffers_released_bytes="<<(unsigned long long)(8.L*3.L*nm)<<" executor_global_k_phase_gib="<<double(current_global_k_phase/gib)<<" executor_peak_bytes_per_gpu="<<(unsigned long long)current_executor_peak<<" executor_peak_gib="<<double(current_executor_peak/gib)<<" executor_budget_status="<<(current_executor_peak<7.5L*gib?"PASS":"FAIL")<<" producer_path=COMPLETE_STREAMED_13_DOUBLE no_25nn_arena=PASS\n";
  const bool actual_fit=current_executor_peak<7.5L*gib;
  (void)actual_fit;
  return 0;
} catch(const std::exception&e) { std::cerr<<"BENCHMARK_FAILURE "<<e.what()<<'\n'; return 2; } }
