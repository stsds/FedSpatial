#pragma once
#include "fedspatial/model.hpp"
#include <cstdint>
#include <functional>
namespace fedspatial {
MaternDerivatives cpu_matern(const Matrix&, const Matrix&, int nu2, double alpha, double range);
LocalComputationResult cpu_local(const WorkerInput&, const Matrix&, const GlobalParameters&,
                                 bool fixed_zero_mean = false);
GammaLocalSummary cpu_gamma(const WorkerInput&, const Matrix&, const GlobalParameters&);
GammaLocalSummary cpu_gamma_from_local(const WorkerInput&, const GlobalParameters&,
                                       const LocalComputationResult&);
PDerivativeWorkspace cpu_derivatives(const WorkerInput&, const Matrix&, const GlobalParameters&);
DeltaThetaLocalResult cpu_delta_theta(const WorkerInput&, const Matrix&, const GlobalParameters&);
DeltaThetaLocalResult cpu_delta_theta_from_local(
    const WorkerInput&, const Matrix&, const GlobalParameters&,
    LocalComputationResult&, bool fixed_zero_mean = false);
void cpu_k_correction(const Matrix&, const GlobalParameters&, Matrix*, Matrix*);
void cpu_update_mu_sigma(GlobalParameters&, const Matrix&, const MuSigmaLocalSummary&, int workers);
PSynchronousRoundResult cpu_newton(const GlobalParameters&, const Matrix&, const Matrix&);
#ifdef FEDSPATIAL_SCALAPACK
// Only the leader evaluates the dense expression graph. Followers service
// operations with distributed buffers, never complete derivative workspaces.
void cpu_worker_execute(const void* communicator, const std::function<void()>&);
bool cpu_distributed_matmul(const Matrix&, const Matrix&, Matrix*);
bool cpu_distributed_matmul3(const Matrix&, const Matrix&, const Matrix&, Matrix*);
bool cpu_distributed_subtract_triples(Matrix&, const Matrix* const*,
                                      const Matrix* const*, const Matrix* const*,
                                      int, Matrix*);
void cpu_distributed_pin(const Matrix&);
void cpu_distributed_unpin(const Matrix&);
bool cpu_distributed_active();
bool cpu_distributed_trace4_symmetric(const Matrix&,const Matrix&,
                                      const Matrix&,const Matrix&,double*);
bool cpu_distributed_build_subtract_triples(
    Matrix&,const Matrix* const*,const Matrix* const*,const Matrix* const*,int,int*);
bool cpu_distributed_matmul_handle(int,const Matrix&,Matrix*);
bool cpu_distributed_trace_product_handle(const Matrix&,int,double*);
void cpu_distributed_scale_handle(int,double);
void cpu_distributed_unpin_handle(int);
int cpu_distributed_release_to_handle(Matrix&);
int cpu_distributed_clone_scale_handle(int,double);
int cpu_distributed_zero_handle(int);
int cpu_distributed_inverse_handle(const Matrix&);
bool cpu_distributed_matmul_handles(int,int,int,int,int,Matrix*);
bool cpu_distributed_trace_product_handles(int,int,double*);
void cpu_distributed_set_site_context(const Matrix&,int,double,double);
bool cpu_distributed_build_site_subtract_triples(
    int,int,const Matrix* const*,const Matrix* const*,const Matrix* const*,int,double,int*);
bool cpu_distributed_solve(const Matrix&, const Matrix&, Matrix*);
bool cpu_distributed_cholesky(const Matrix&, Matrix*);
#endif
// Row ownership is contiguous and independent of thread count. Each output
// element retains the original scalar summation order.
int cpu_row_begin(int rows);
int cpu_row_end(int rows);
void cpu_gather_rows(Matrix&);
// Logical algorithm payload, excluding measurement-only reductions. The
// process-local counter is summed across MPI_COMM_WORLD by the driver.
void cpu_reset_communication_bytes();
void cpu_add_communication_bytes(std::uint64_t);
std::uint64_t cpu_communication_bytes();
#ifdef FEDSPATIAL_CPU_MPI
void cpu_set_worker_comm(int rank, int size, const void* communicator);
#endif
}
