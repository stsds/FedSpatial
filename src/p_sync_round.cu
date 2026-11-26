#include "fedspatial/gpu_context.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <sstream>
#include <system_error>
#include <thread>
#include <typeinfo>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <mutex>
#include <cstdlib>
#include <fstream>
#include <iomanip>

namespace fedspatial {
namespace {
// These records deliberately use stderr and endl: a timeout must not discard
// the last completed boundary behind a buffered stdout stream.
std::mutex stage_log_mutex;
void stage_log(int round, const char* stage, const char* event,
               int worker = -1, int device = -1) {
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  std::lock_guard<std::mutex> lock(stage_log_mutex);
  std::cerr << "NWORKER_STAGE ts_ms=" << ms << " round=" << round
            << " stage=" << stage << " event=" << event;
  if (worker >= 0) std::cerr << " worker=" << worker;
  if (device >= 0) std::cerr << " device=" << device;
  std::cerr << std::endl;
}
bool stage_cuda_timing_enabled() {
  const char* value = std::getenv("FEDSPATIAL_STAGE_CUDA_TIMING");
  return value && value[0] != '\0' && value[0] != '0';
}
void check_round_cuda(cudaError_t status);
void stage_timing_log(int round, const char* stage, int worker, int device,
                      double wall_ms, float device_ms) {
  std::lock_guard<std::mutex> lock(stage_log_mutex);
  std::cerr << "NWORKER_STAGE_TIMING round=" << round << " stage=" << stage
            << " worker=" << worker << " device=" << device
            << " wall_ms=" << wall_ms << " cuda_ms=" << device_ms << std::endl;
}
// Test-only transport audit.  This executes after the local device producer
// has completed but before the compact peer aggregation consumes its 13
// values.  It is intentionally opt-in: the normal round retains its
// device-only transport boundary.
void dump_delta_theta_preaggregation_audit(
    const std::vector<DeviceDeltaThetaSummary>& summaries,
    const std::vector<std::unique_ptr<GpuContext>>& contexts, int round) {
  const char* path = std::getenv("FEDSPATIAL_DELTA_THETA_PREAGG_AUDIT");
  if (!path || !path[0]) return;
  std::ofstream output(path, std::ios::app);
  if (!output) throw std::runtime_error("cannot open FEDSPATIAL_DELTA_THETA_PREAGG_AUDIT");
  output << std::setprecision(17);
  for (size_t worker = 0; worker < summaries.size(); ++worker) {
    std::array<double,13> values{};
    check_round_cuda(cudaSetDevice(contexts[worker]->device_id()));
    check_round_cuda(cudaMemcpyAsync(values.data(), summaries[worker].values,
                                     values.size()*sizeof(double), cudaMemcpyDeviceToHost,
                                     contexts[worker]->stream()));
    check_round_cuda(cudaStreamSynchronize(contexts[worker]->stream()));
    output << "round=" << round << " worker=" << worker
           << " logical_device=" << contexts[worker]->device_id() << " values=";
    for (double value : values) output << value << ',';
    output << '\n';
  }
  output.flush();
}
void dump_after_gamma_audit(const GlobalParameters& state, int round) {
  const char* path = std::getenv("FEDSPATIAL_DELTA_THETA_PREAGG_AUDIT");
  if (!path || !path[0]) return;
  // Fixed host order makes this a useful upstream-state fingerprint without
  // turning the audit into a 400-by-400 dump.
  double mu_sum=0., mu_weighted=0., sigma_sum=0., sigma_weighted=0.;
  for (size_t i=0;i<state.mu.a.size();++i) {
    mu_sum += state.mu.a[i]; mu_weighted += double(i+1)*state.mu.a[i];
  }
  for (size_t i=0;i<state.sigma.a.size();++i) {
    sigma_sum += state.sigma.a[i]; sigma_weighted += double(i+1)*state.sigma.a[i];
  }
  std::ofstream output(path, std::ios::app);
  if (!output) throw std::runtime_error("cannot open FEDSPATIAL_DELTA_THETA_PREAGG_AUDIT");
  output << std::setprecision(17) << "round=" << round << " after_gamma gamma=";
  for (double value : state.gamma.a) output << value << ',';
  output << " mu_sum=" << mu_sum << " mu_weighted=" << mu_weighted
         << " sigma_sum=" << sigma_sum << " sigma_weighted=" << sigma_weighted << '\n';
  output.flush();
}
void dump_device_audit(GpuContext& context, const char* label, const double* device, size_t count) {
  const char* path = std::getenv("FEDSPATIAL_DELTA_THETA_PREAGG_AUDIT");
  if (!path || !path[0]) return;
  std::vector<double> host(count);
  check_round_cuda(cudaMemcpyAsync(host.data(), device, count*sizeof(double), cudaMemcpyDeviceToHost, context.stream()));
  check_round_cuda(cudaStreamSynchronize(context.stream()));
  double sum=0., weighted=0.; for(size_t i=0;i<count;++i){sum+=host[i];weighted+=double(i+1)*host[i];}
  std::ofstream output(path,std::ios::app); if(!output) throw std::runtime_error("cannot open FEDSPATIAL_DELTA_THETA_PREAGG_AUDIT");
  output<<std::setprecision(17)<<label<<" device="<<context.device_id()<<" sum="<<sum<<" weighted="<<weighted<<'\n'; output.flush();
  // The text fingerprint above is cheap to scan.  The optional sidecar is a
  // test-only exact checkpoint used to identify the first differing element
  // across fresh processes; it never participates in model arithmetic.
  if (const char* exact = std::getenv("FEDSPATIAL_GLOBAL_SOLVE_AUDIT_EXACT"); exact && exact[0] && exact[0] != '0') {
    const std::string sidecar=std::string(path)+"."+label+".fp64.bin";
    std::ofstream binary(sidecar,std::ios::binary|std::ios::trunc);
    if(!binary) throw std::runtime_error("cannot open global solve audit sidecar");
    binary.write(reinterpret_cast<const char*>(host.data()),std::streamsize(count*sizeof(double)));
    if(!binary) throw std::runtime_error("cannot write global solve audit sidecar");
  }
}
void check_round_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check_round_cuda(cusolverStatus_t status) {
  if (status != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER P round failure");
}
const char* round_cublas_status_name(cublasStatus_t status) {
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
void check_round_blas(GpuContext& context, cublasStatus_t status, const char* operation,
                      int m, int n, int k, int lda, int ldb, int ldc,
                      size_t a_capacity, size_t b_capacity, size_t c_capacity) {
  std::ostringstream detail;
  detail << "cuBLAS P round " << operation << " status=" << round_cublas_status_name(status)
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
// Stream order supplies dependencies for the global-K GEMM chain.  Keep its
// error check asynchronous; the enclosing compact checkpoint synchronizes.
void check_round_blas_enqueue(GpuContext& context, cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cuBLAS enqueue failed for ") + operation +
                             " on device " + std::to_string(context.device_id()));
}
void check_gamma_potrf_info(GpuContext& context, int info, int q, size_t xx_capacity,
                            size_t work_capacity) {
  if (!info) return;
  std::ostringstream detail;
  detail << "gamma global POTRF failed info=" << info << " device=" << context.device_id()
         << " n=" << q << " lda=" << q << " xx_capacity=" << xx_capacity
         << " work_capacity=" << work_capacity << " doubles";
  throw std::runtime_error(detail.str());
}

// One thread owns each output element and visits worker slots in increasing
// logical-worker ID.  This intentionally trades tiny-summary parallelism for
// a fixed FP64 addition order (and byte-repeatable results).
__global__ void reduce_compact_mu_sigma(const double* slots, double* mean,
                                        int workers, int m, double delta) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = m + m * m;
  if (index >= count) return;
  double value = 0.;
  for (int worker = 0; worker < workers; ++worker)
    value += slots[worker * count + index];
  // Local device state stores B'M^-1e and B'M^-1B.  Preserve the historical
  // sufficient-statistic convention here, on GPU, before the global solve.
  mean[index] = (index < m ? -value / delta : value / delta) / workers;
}

// The server update is intentionally scalar-owned: its loop order is the
// historical host Matrix order, while all arithmetic stays in FP64 on the
// aggregation GPU.  This also makes the stored covariance explicitly
// symmetric before it is consumed by the next worker stage.
// This must not symmetrize in the same launch as the add.  The old fused
// kernel let a lower-triangle thread read precision[j,i] while the distinct
// upper-triangle thread was still updating it: CUDA provides no grid-wide
// ordering inside a kernel.  Keep the elementwise add parallel, then use the
// existing separate symmetrization launch as the stream-ordered boundary.
__global__ void add_precision(double* precision, const double* mean_sigma,
                              double scale, int n) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * n) return;
  precision[t] += scale * mean_sigma[t];
}
__global__ void fixed_mu_from_sigma(const double* sigma, const double* mean,
                                    double* mu, double scale, int n) {
  if (blockIdx.x || threadIdx.x) return;
  for (int i=0;i<n;++i) {
    double value=0.;
    for (int j=0;j<n;++j) value += sigma[i*n+j]*mean[j];
    mu[i]=scale*value;
  }
}
__global__ void set_identity_device(double* value, int n) {
  const int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t<n*n) value[t]=(t/n==t%n)?1.:0.;
}
__global__ void symmetrize_matrix(double* value, int n) {
  const int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=n*n) return;
  const int i=t/n,j=t%n; if(j>i) return;
  const double v=(i==j?value[t]:.5*(value[i*n+j]+value[j*n+i]));
  value[i*n+j]=v; value[j*n+i]=v;
}

__global__ void reduce_compact_gamma(const double* slots, double* mean,
                                     int workers, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  double value = 0.;
  for (int worker = 0; worker < workers; ++worker) value += slots[worker * count + index];
  mean[index] = value / workers;
}
__global__ void initialize_compact_slots(double* slots, int count, int workers) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if (index >= count) return;
  const double pattern=.125*double(index%13);
  for (int worker=0; worker<workers; ++worker)
    slots[worker*count+index]=double(worker+1)+pattern;
}

// The delta/theta summary is deliberately reduced by one thread in worker-ID
// order.  A 13-double slot is tiny, but this order is part of the numerical
// contract: it makes repeated rounds byte-repeatable across transport modes.
__global__ void reduce_compact_delta_theta(const double* slots, double* total,
                                           int workers) {
  if (blockIdx.x || threadIdx.x) return;
  for (int i = 0; i < 13; ++i) {
    double value = 0.;
    for (int worker = 0; worker < workers; ++worker) value += slots[13 * worker + i];
    total[i] = value;
  }
}

// Scalar, fixed-order K inverse. This mirrors the legacy Cholesky solve
// order, but executes entirely on the designated aggregation GPU.
__global__ void inverse_spd_fixed(const double* k, double* l, double* inverse, int n) {
  if (blockIdx.x || threadIdx.x) return;
  for (int i=0;i<n*n;++i) { l[i]=0.; inverse[i]=0.; }
  for (int i=0;i<n;++i) for(int j=0;j<=i;++j) {
    double s=k[i*n+j]; for(int q=0;q<j;++q) s-=l[i*n+q]*l[j*n+q];
    l[i*n+j]=(i==j?sqrt(s):s/l[j*n+j]);
  }
  // Columns are solved in the same forward/backward order as solve_spd.
  for(int c=0;c<n;++c) {
    for(int i=0;i<n;++i) { double s=(i==c?1.:0.); for(int q=0;q<i;++q)s-=l[i*n+q]*inverse[q*n+c]; inverse[i*n+c]=s/l[i*n+i]; }
    for(int i=n-1;i>=0;--i) { double s=inverse[i*n+c]; for(int q=i+1;q<n;++q)s-=l[q*n+i]*inverse[q*n+c]; inverse[i*n+c]=s/l[i*n+i]; }
  }
}

bool use_scalar_global_inverse_reference() {
  const char* value=std::getenv("FEDSPATIAL_GLOBAL_SCALAR_INVERSE");
  return value && value[0] != '\0' && value[0] != '0';
}

// Forms A^-1 using a GPU-parallel POTRF plus two matrix-RHS TRSM calls.  The
// caller supplies storage already owned by the active global phase, so this
// has no persistent m-by-m memory increment.  `factor` may be an existing
// valid Cholesky factor (the K factor constructed with B).
void inverse_spd_parallel(GpuContext& context, const double* source, double* factor,
                          double* inverse, int n, bool factor_is_valid) {
  if (use_scalar_global_inverse_reference()) {
    inverse_spd_fixed<<<1,1,0,context.stream()>>>(source,factor,inverse,n);
    check_round_cuda(cudaGetLastError());
    return;
  }
  auto& region=const_cast<RegionGpuState&>(context.region());
  if (!factor_is_valid) {
    check_round_cuda(cudaMemcpyAsync(factor,source,size_t(n)*n*sizeof(double),
                                     cudaMemcpyDeviceToDevice,context.stream()));
    int work_size=0;
    check_round_cuda(cusolverDnDpotrf_bufferSize(context.cusolver(),CUBLAS_FILL_MODE_LOWER,
                                                  n,factor,n,&work_size));
    if (region.potrf_work_capacity < size_t(work_size)) {
      if (region.potrf_work) check_round_cuda(cudaFree(region.potrf_work));
      check_round_cuda(cudaMalloc(&region.potrf_work,size_t(work_size)*sizeof(double)));
      region.potrf_work_capacity=work_size;
    }
    check_round_cuda(cusolverDnDpotrf(context.cusolver(),CUBLAS_FILL_MODE_LOWER,n,factor,n,
                                       region.potrf_work,work_size,region.potrf_info));
    int info=-1;
    check_round_cuda(cudaMemcpyAsync(&info,region.potrf_info,sizeof(int),cudaMemcpyDeviceToHost,
                                     context.stream()));
    check_round_cuda(cudaStreamSynchronize(context.stream()));
    if (info) throw std::runtime_error("global SPD POTRF failed info="+std::to_string(info));
  }
  set_identity_device<<<(n*n+255)/256,256,0,context.stream()>>>(inverse,n);
  check_round_cuda(cudaGetLastError());
  const double one=1.;
  check_round_blas_enqueue(context,cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,
      CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,n,&one,factor,n,inverse,n),
      "DTRSM-global-inverse(N)");
  check_round_blas_enqueue(context,cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,
      CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,n,&one,factor,n,inverse,n),
      "DTRSM-global-inverse(T)");
}

// Small scalar reductions deliberately have one owner.  GEMMs below create
// the same ordered intermediates as the legacy matmul expressions; these
// kernels only perform their final trace/quadratic contractions.
__global__ void global_k_u(const double* w, const double* mu, double* u, int n) {
  if (blockIdx.x || threadIdx.x) return;
  for (int i=0;i<n;++i) { double v=0.; for (int j=0;j<n;++j) v+=w[i*n+j]*mu[j]; u[i]=v; }
}
__global__ void global_k_gradient_add(const double* wki, const double* ki,
                                      const double* wkiw, const double* sigma,
                                      const double* u, double* total, int slot, int n) {
  if (blockIdx.x || threadIdx.x) return;
  double a=0., b=0., c=0.;
  for(int i=0;i<n;++i) for(int j=0;j<n;++j) { a+=(i==j?wki[i*n+j]:0.); b+=u[i]*ki[i*n+j]*u[j]; c+=wkiw[i*n+j]*sigma[j*n+i]; }
  total[slot]+=a-b-c;
}
__global__ void global_k_hessian_add(const double* w, const double* kij,
    const double* wkjwki, const double* wkijw, const double* kjwki,
    const double* kiwkj, const double* wkjwkiw, const double* wkiwkjw,
    const double* sigma, const double* u, double* total, int slot, int n) {
  if (blockIdx.x || threadIdx.x) return;
  double a=0.,b=0.,c=0.,d=0.,e=0.,f=0.,g=0.,h=0.;
  for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
    // trace(W Kij) is a full contraction, unlike trace(W Kj W Ki),
    // whose product has already been materialized in wkjwki.
    a+=w[i*n+j]*kij[j*n+i]; b+=wkjwki[i*n+j]*(i==j?1.:0.);
    c+=u[i]*kij[i*n+j]*u[j]; d+=u[i]*kjwki[i*n+j]*u[j]; e+=u[i]*kiwkj[i*n+j]*u[j];
    f+=wkjwkiw[i*n+j]*sigma[j*n+i]; g+=wkiwkjw[i*n+j]*sigma[j*n+i]; h+=wkijw[i*n+j]*sigma[j*n+i];
  }
  total[slot]+=a-b-c+d+e+f+g-h;
}
__global__ void global_k_mirror(double* total) { if (!blockIdx.x && !threadIdx.x) { total[4+3*2+1]=total[4+3*1+2]; } }
// The central correction owns these five m-by-m K derivatives.  In
// particular, it must not trigger the worker-local combined B/A/M bundles.
__global__ void global_k_derivatives(const double* knots,int m,int nu2,double alpha,
                                     double range,double* ka,double* kr,double* kaa,
                                     double* kar,double* krr) {
  const int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=m*m)return;
  const int i=t/m,j=t%m; const double dx=knots[2*i]-knots[2*j],dy=knots[2*i+1]-knots[2*j+1];
  double u=sqrt(dx*dx+dy*dy)/range,e,dr,drr;
  if(nu2==1){e=exp(-u);ka[t]=e;dr=e*u/range;drr=alpha*e*(u*u-2.*u)/(range*range);}
  else if(nu2==3){u*=sqrt(3.);e=exp(-u);ka[t]=(1.+u)*e;dr=e*u*u/range;drr=alpha*e*(u*u*u-3.*u*u)/(range*range);}
  else {u*=sqrt(5.);e=exp(-u);ka[t]=(1.+u+u*u/3.)*e;dr=e*(u*u+u*u*u)/(3.*range);drr=alpha*e*(u*u*u*u-3.*u*u*u-3.*u*u)/(3.*range*range);}
  kr[t]=alpha*dr;kaa[t]=0.;kar[t]=dr;krr[t]=drr;
}

void global_k_product(GpuContext& c, const double* left, const double* right,
                      double* out, int n) {
  const double one=1., zero=0.;
  // Row-major out=left*right; cuBLAS views all three buffers transposed.
  check_round_blas_enqueue(c, cublasDgemm(c.cublas(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                    &one, right, n, left, n, &zero, out, n), "DGEMM-global-K");
}

__device__ double dtr(const double* a,int n) { double x=0.;for(int i=0;i<n;++i)x+=a[i*n+i];return x; }
__device__ double dtrace_product_n(const double* a,const double* b,int n) { double x=0.;for(int i=0;i<n;++i)for(int j=0;j<n;++j)x+=a[i*n+j]*b[j*n+i];return x; }
__device__ double dquad_n(const double* u,const double* a,const double* v,int n) { double x=0.;for(int i=0;i<n;++i){double y=0.;for(int j=0;j<n;++j)y+=a[i*n+j]*v[j];x+=u[i]*y;}return x; }
__device__ double dtr_abc(const double* a,const double* b,const double* c,int n) { double x=0.;for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int k=0;k<n;++k)x+=a[i*n+j]*b[j*n+k]*c[k*n+i];return x; }
__device__ double dquad_abc(const double* u,const double* a,const double* b,const double* c,const double* v,int n) { double x=0.;for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int k=0;k<n;++k)for(int z=0;z<n;++z)x+=u[i]*a[i*n+j]*b[j*n+k]*c[k*n+z]*v[z];return x; }

// The compact 13-double record has already received the global-K correction.
// One FP64 GPU thread reproduces the legacy 3x3 Jacobi/eigen-modification,
// Cholesky solve, and positivity loop in a fixed instruction/order sequence.
__global__ void global_newton_update(const double* total, double delta,
                                     double alpha, double range, double* result) {
  if(blockIdx.x||threadIdx.x)return;
  double g[3]={total[1],total[2],total[3]};
  double a[9],v[9]={1,0,0,0,1,0,0,0,1};
  // The legacy rule operates on a symmetric Hessian.  Make the symmetry
  // explicit and deterministic before its Jacobi sweeps.
  for(int i=0;i<3;++i) for(int j=0;j<3;++j)
    a[3*i+j]=(i==j ? total[4+3*i+j] : .5*(total[4+3*i+j]+total[4+3*j+i]));
  for(int it=0;it<64;++it){int p=0,q=1;double mx=fabs(a[1]);for(int i=0;i<3;++i)for(int j=i+1;j<3;++j)if(fabs(a[3*i+j])>mx){mx=fabs(a[3*i+j]);p=i;q=j;}if(mx<1e-15)break;double angle=.5*atan2(2.*a[3*p+q],a[3*q+q]-a[3*p+p]),c=cos(angle),s=sin(angle);for(int k=0;k<3;++k){double pk=a[3*p+k],qk=a[3*q+k];a[3*p+k]=c*pk-s*qk;a[3*q+k]=s*pk+c*qk;}for(int k=0;k<3;++k){double kp=a[3*k+p],kq=a[3*k+q];a[3*k+p]=c*kp-s*kq;a[3*k+q]=s*kp+c*kq;double vp=v[3*k+p],vq=v[3*k+q];v[3*k+p]=c*vp-s*vq;v[3*k+q]=s*vp+c*vq;}}
  int order[3]={0,1,2};for(int i=0;i<3;++i)for(int j=i+1;j<3;++j)if(a[3*order[i]+order[i]]>a[3*order[j]+order[j]]){int z=order[i];order[i]=order[j];order[j]=z;} double mod[9]={};for(int k=0;k<3;++k){int ix=order[k];double d=fmax(fabs(a[3*ix+ix]),.01);for(int i=0;i<3;++i)for(int j=0;j<3;++j)mod[3*i+j]+=v[3*i+ix]*d*v[3*j+ix];}
  double l[9]={},y[3],dir[3];for(int i=0;i<3;++i)for(int j=0;j<=i;++j){double s=mod[3*i+j];for(int q=0;q<j;++q)s-=l[3*i+q]*l[3*j+q];l[3*i+j]=(i==j?sqrt(s):s/l[3*j+j]);}for(int i=0;i<3;++i){double s=g[i];for(int q=0;q<i;++q)s-=l[3*i+q]*y[q];y[i]=s/l[3*i+i];}for(int i=2;i>=0;--i){double s=y[i];for(int q=i+1;q<3;++q)s-=l[3*q+i]*dir[q];dir[i]=s/l[3*i+i];} double step=1.;int halves=0;while(delta-step*dir[0]<=0.||alpha-step*dir[1]<=0.||range-step*dir[2]<=0.){step*=.5;++halves;}result[0]=delta-step*dir[0];result[1]=alpha-step*dir[1];result[2]=range-step*dir[2];result[3]=step;result[4]=double(halves);
}

// Fixed worker-ID slots make both copies and the FP64 summation order stable.
// The sole host fallback is a counted pinned relay for an unsupported compact
// peer link; raw worker arrays never enter this function.
void gather_compact_gamma(NWorkerPSynchronousExecutor* executor,
                          const std::vector<DeviceGammaLocalSummary>& local, int q,
                          double** mean_gamma, double** mean_xx) {
  const int central = executor->contexts_[0]->device_id(), workers = int(local.size());
  const size_t count = size_t(q) + size_t(q) * q, bytes = count * sizeof(double);
  check_round_cuda(cudaSetDevice(central));
  if (executor->compact_aggregation_capacity_ < count * size_t(workers + 1)) {
    if (executor->compact_aggregation_) check_round_cuda(cudaFree(executor->compact_aggregation_));
    check_round_cuda(cudaMalloc(&executor->compact_aggregation_, count * size_t(workers + 1) * sizeof(double)));
    executor->compact_aggregation_capacity_ = count * size_t(workers + 1);
  }
  if (executor->pinned_compact_capacity_ < bytes) {
    if (executor->pinned_compact_) check_round_cuda(cudaFreeHost(executor->pinned_compact_));
    check_round_cuda(cudaMallocHost(&executor->pinned_compact_, bytes)); executor->pinned_compact_capacity_ = bytes;
  }
  double* slots = executor->compact_aggregation_;
  for (int worker = 0; worker < workers; ++worker) {
    double* dst = slots + size_t(worker) * count;
    const int source = executor->contexts_[worker]->device_id();
    const size_t gamma_bytes = size_t(q) * sizeof(double), xx_bytes = size_t(q) * q * sizeof(double);
    if (source == central) {
      check_round_cuda(cudaMemcpyAsync(dst, local[worker].gamma, gamma_bytes, cudaMemcpyDeviceToDevice,
                                       executor->contexts_[0]->stream()));
      check_round_cuda(cudaMemcpyAsync(dst + q, local[worker].xx, xx_bytes, cudaMemcpyDeviceToDevice,
                                       executor->contexts_[0]->stream()));
    } else {
      int peer = 0; check_round_cuda(cudaDeviceCanAccessPeer(&peer, central, source));
      if (peer) {
        const cudaError_t enabled = cudaDeviceEnablePeerAccess(source, 0);
        if (enabled != cudaSuccess && enabled != cudaErrorPeerAccessAlreadyEnabled) check_round_cuda(enabled);
        if (enabled == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
        check_round_cuda(cudaMemcpyPeerAsync(dst, central, local[worker].gamma, source, gamma_bytes,
                                              executor->contexts_[0]->stream()));
        check_round_cuda(cudaMemcpyPeerAsync(dst + q, central, local[worker].xx, source, xx_bytes,
                                              executor->contexts_[0]->stream()));
        executor->peer_summary_bytes_ += static_cast<long long>(bytes);
      } else {
        auto* pinned = static_cast<double*>(executor->pinned_compact_);
        check_round_cuda(cudaSetDevice(source));
        check_round_cuda(cudaMemcpyAsync(pinned, local[worker].gamma, gamma_bytes, cudaMemcpyDeviceToHost,
                                         executor->contexts_[worker]->stream()));
        check_round_cuda(cudaMemcpyAsync(pinned + q, local[worker].xx, xx_bytes, cudaMemcpyDeviceToHost,
                                         executor->contexts_[worker]->stream()));
        check_round_cuda(cudaStreamSynchronize(executor->contexts_[worker]->stream()));
        check_round_cuda(cudaSetDevice(central));
        check_round_cuda(cudaMemcpyAsync(dst, pinned, bytes, cudaMemcpyHostToDevice,
                                         executor->contexts_[0]->stream()));
        executor->pinned_summary_bytes_ += static_cast<long long>(bytes);
      }
    }
  }
  double* mean = slots + size_t(workers) * count;
  constexpr int threads = 256;
  reduce_compact_gamma<<<static_cast<unsigned>((count + threads - 1) / threads), threads, 0, executor->contexts_[0]->stream()>>>(slots, mean, workers, int(count));
  check_round_cuda(cudaGetLastError());
  *mean_gamma = mean; *mean_xx = mean + q;
}

void solve_and_broadcast_gamma(NWorkerPSynchronousExecutor* executor, double* rhs, double* xx, int q) {
  GpuContext& central_context = *executor->contexts_[0]; const int central = central_context.device_id();
  check_round_cuda(cudaSetDevice(central));
  auto& region = const_cast<RegionGpuState&>(central_context.region());
  int work_size = 0; check_round_cuda(cusolverDnDpotrf_bufferSize(central_context.cusolver(), CUBLAS_FILL_MODE_LOWER, q, xx, q, &work_size));
  if (region.potrf_work_capacity < size_t(work_size)) throw std::runtime_error("gamma solve workspace capacity");
  check_round_cuda(cusolverDnDpotrf(central_context.cusolver(), CUBLAS_FILL_MODE_LOWER, q, xx, q, region.potrf_work, work_size, region.potrf_info));
  const double one = 1.;
  check_round_blas(central_context, cublasDtrsm(central_context.cublas(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, q, 1, &one, xx, q, rhs, q), "DTRSM-gamma(N)", q, 1, 0, q, q, q, size_t(q)*q, size_t(q), size_t(q));
  check_round_blas(central_context, cublasDtrsm(central_context.cublas(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, q, 1, &one, xx, q, rhs, q), "DTRSM-gamma(T)", q, 1, 0, q, q, q, size_t(q)*q, size_t(q), size_t(q));
  int info = -1; check_round_cuda(cudaMemcpyAsync(&info, region.potrf_info, sizeof(int), cudaMemcpyDeviceToHost, central_context.stream()));
  check_round_cuda(cudaStreamSynchronize(central_context.stream()));
  check_gamma_potrf_info(central_context, info, q, size_t(q)*q, region.potrf_work_capacity);
  for (int worker = 0; worker < executor->worker_count(); ++worker) {
    GpuContext& target_context = *executor->contexts_[worker]; auto& target = const_cast<RegionGpuState&>(target_context.region());
    const int target_device = target_context.device_id();
    if (target_device == central) check_round_cuda(cudaMemcpyAsync(target.parameter_gamma, rhs, size_t(q)*sizeof(double), cudaMemcpyDeviceToDevice, central_context.stream()));
    else { int peer = 0; check_round_cuda(cudaDeviceCanAccessPeer(&peer, target_device, central));
      if (peer) { check_round_cuda(cudaSetDevice(target_device)); const cudaError_t enabled=cudaDeviceEnablePeerAccess(central,0); if(enabled!=cudaSuccess&&enabled!=cudaErrorPeerAccessAlreadyEnabled)check_round_cuda(enabled); if(enabled==cudaErrorPeerAccessAlreadyEnabled)(void)cudaGetLastError(); check_round_cuda(cudaMemcpyPeerAsync(target.parameter_gamma,target_device,rhs,central,size_t(q)*sizeof(double),target_context.stream())); executor->peer_summary_bytes_ += static_cast<long long>(q)*sizeof(double); }
      else { auto* pinned=static_cast<double*>(executor->pinned_compact_); check_round_cuda(cudaSetDevice(central)); check_round_cuda(cudaMemcpyAsync(pinned,rhs,size_t(q)*sizeof(double),cudaMemcpyDeviceToHost,central_context.stream())); check_round_cuda(cudaStreamSynchronize(central_context.stream())); check_round_cuda(cudaSetDevice(target_device)); check_round_cuda(cudaMemcpyAsync(target.parameter_gamma,pinned,size_t(q)*sizeof(double),cudaMemcpyHostToDevice,target_context.stream())); executor->pinned_summary_bytes_ += static_cast<long long>(q)*sizeof(double); }
    }
  }
  for (const auto& context : executor->contexts_) {
    check_round_cuda(cudaSetDevice(context->device_id()));
    check_round_cuda(cudaStreamSynchronize(context->stream()));
    check_round_cuda(cudaGetLastError());
  }
}

// Moves only m + m*m doubles (mu/sigma sufficient statistics).  In
// particular, no n-by-n/n-by-m region buffer is accepted by this routine.
void gather_compact_mu_sigma(NWorkerPSynchronousExecutor* executor,
                             const std::vector<DeviceMuSigmaLocalState>& local,
                             int m, double delta, double** device_mean_mu,
                             double** device_mean_sigma) {
  const auto check_compact = [](cudaError_t status, const char* call) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string("gather_compact_mu_sigma ") + call +
                             " code=" + std::to_string(static_cast<int>(status)) +
                             " (" + cudaGetErrorString(status) + ")");
  };
  const int central_device = executor->contexts_[0]->device_id();
  const int workers = static_cast<int>(local.size());
  const size_t count = size_t(m) + size_t(m) * m;
  const size_t bytes = count * sizeof(double);
  const size_t required = count * (workers + 1);
  check_compact(cudaSetDevice(central_device), "cudaSetDevice(central)");
  if (executor->compact_aggregation_capacity_ < required) {
    if (executor->compact_aggregation_) check_compact(cudaFree(executor->compact_aggregation_), "cudaFree(compact)");
    check_compact(cudaMalloc(&executor->compact_aggregation_, required * sizeof(double)), "cudaMalloc(compact)");
    executor->compact_aggregation_capacity_ = required;
  }
  if (executor->pinned_compact_capacity_ < bytes) {
    if (executor->pinned_compact_) check_compact(cudaFreeHost(executor->pinned_compact_), "cudaFreeHost(compact)");
    check_compact(cudaMallocHost(&executor->pinned_compact_, bytes), "cudaMallocHost(compact)");
    executor->pinned_compact_capacity_ = bytes;
  }
  double* slots = executor->compact_aggregation_;
  for (int worker = 0; worker < workers; ++worker) {
    const int source_device = executor->contexts_[worker]->device_id();
    double* destination = slots + size_t(worker) * count;
    // run_local joins host threads, not their asynchronous CUDA streams.
    // In particular, a peer DMA from worker 1 otherwise has no dependency on
    // that worker's bt_minv_e/bt_minv_B producer stream and may consume a
    // partially written sufficient statistic.  This is the same explicit
    // producer/consumer boundary used by the compact delta/theta gather.
    check_compact(cudaSetDevice(source_device), "cudaSetDevice(local producer)");
    check_compact(cudaStreamSynchronize(executor->contexts_[worker]->stream()),
                  "cudaStreamSynchronize(local producer)");
    if (const char* audit = std::getenv("FEDSPATIAL_DELTA_THETA_PREAGG_AUDIT"); audit && audit[0]) {
      // Test-only ownership check: download after the exact producer-stream
      // fence above, then fingerprint in fixed host order.  It is deliberately
      // not a global device synchronization and is never enabled in normal
      // execution.
      std::vector<double> host(count);
      check_compact(cudaMemcpyAsync(host.data(), local[worker].bt_minv_e,
                                    size_t(m)*sizeof(double), cudaMemcpyDeviceToHost,
                                    executor->contexts_[worker]->stream()),
                    "cudaMemcpyAsync(local mu audit)");
      check_compact(cudaMemcpyAsync(host.data()+m, local[worker].bt_minv_B,
                                    size_t(m)*m*sizeof(double), cudaMemcpyDeviceToHost,
                                    executor->contexts_[worker]->stream()),
                    "cudaMemcpyAsync(local sigma audit)");
      check_compact(cudaStreamSynchronize(executor->contexts_[worker]->stream()),
                    "cudaStreamSynchronize(local audit)");
      double sum=0., weighted=0.;
      for (size_t i=0;i<host.size();++i) { sum += host[i]; weighted += double(i+1)*host[i]; }
      std::ofstream output(audit, std::ios::app);
      if (!output) throw std::runtime_error("cannot open FEDSPATIAL_DELTA_THETA_PREAGG_AUDIT");
      output << std::setprecision(17) << "mu_sigma_local worker=" << worker
             << " device=" << source_device << " sum=" << sum
             << " weighted=" << weighted << '\n';
      output.flush();
    }
    check_compact(cudaSetDevice(central_device), "cudaSetDevice(central after producer)");
    if (source_device == central_device) {
      check_compact(cudaMemcpyAsync(destination, local[worker].bt_minv_e,
                                       size_t(m) * sizeof(double), cudaMemcpyDeviceToDevice,
                                       executor->contexts_[0]->stream()), "cudaMemcpyAsync(local mu)");
      check_compact(cudaMemcpyAsync(destination + m, local[worker].bt_minv_B,
                                       size_t(m) * m * sizeof(double), cudaMemcpyDeviceToDevice,
                                       executor->contexts_[0]->stream()), "cudaMemcpyAsync(local sigma)");
      continue;
    }
    int can_peer = 0;
    check_compact(cudaDeviceCanAccessPeer(&can_peer, central_device, source_device), "cudaDeviceCanAccessPeer(destination, source)");
    if (can_peer) {
      // cudaMemcpyPeerAsync requires the destination context to have enabled
      // access; doing this lazily also keeps unsupported pairs on the pinned
      // relay path below.
      check_compact(cudaSetDevice(central_device), "cudaSetDevice(peer destination)");
      const cudaError_t enable = cudaDeviceEnablePeerAccess(source_device, 0);
      if (enable != cudaSuccess && enable != cudaErrorPeerAccessAlreadyEnabled)
        check_compact(enable, "cudaDeviceEnablePeerAccess(source)");
      if (enable == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
      check_compact(cudaMemcpyPeerAsync(destination, central_device, local[worker].bt_minv_e,
                                            source_device, size_t(m) * sizeof(double),
                                            executor->contexts_[0]->stream()), "cudaMemcpyPeerAsync(mu)");
      check_compact(cudaMemcpyPeerAsync(destination + m, central_device, local[worker].bt_minv_B,
                                            source_device, size_t(m) * m * sizeof(double),
                                            executor->contexts_[0]->stream()), "cudaMemcpyPeerAsync(sigma)");
      executor->peer_summary_bytes_ += static_cast<long long>(bytes);
    } else {
      // Hardware transport limitation (e.g. RTX4000 <-> GV100): this is an
      // explicit pinned-host relay for compact summaries only, never raw data
      // or model-sized arrays.
      auto* pinned = static_cast<double*>(executor->pinned_compact_);
      check_compact(cudaSetDevice(source_device), "cudaSetDevice(fallback source)");
      check_compact(cudaMemcpyAsync(pinned, local[worker].bt_minv_e,
                                       size_t(m) * sizeof(double), cudaMemcpyDeviceToHost,
                                       executor->contexts_[worker]->stream()), "cudaMemcpyAsync(fallback mu)");
      check_compact(cudaMemcpyAsync(pinned + m, local[worker].bt_minv_B,
                                       size_t(m) * m * sizeof(double), cudaMemcpyDeviceToHost,
                                       executor->contexts_[worker]->stream()), "cudaMemcpyAsync(fallback sigma)");
      check_compact(cudaStreamSynchronize(executor->contexts_[worker]->stream()), "cudaStreamSynchronize(fallback source)");
      check_compact(cudaSetDevice(central_device), "cudaSetDevice(fallback destination)");
      check_compact(cudaMemcpyAsync(destination, pinned, bytes, cudaMemcpyHostToDevice,
                                       executor->contexts_[0]->stream()), "cudaMemcpyAsync(fallback relay)");
      executor->pinned_summary_bytes_ += static_cast<long long>(bytes);
    }
  }
  double* mean = slots + size_t(workers) * count;
  // Local CUDA work ran on worker threads.  cudaGetLastError state is host-
  // thread-local, so the aggregation thread can retain an unrelated status
  // when every logical worker uses the central device.  Start the compact
  // launch boundary clean; the checked stream synchronization below still
  // reports every copy or kernel execution error from this gather.
  (void)cudaGetLastError();
  constexpr int threads = 256;
  reduce_compact_mu_sigma<<<static_cast<unsigned>((count + threads - 1) / threads), threads, 0,
                             executor->contexts_[0]->stream()>>>(slots, mean, workers, m, delta);
  check_compact(cudaGetLastError(), "reduce_compact_mu_sigma launch");
  if (const char* audit = std::getenv("FEDSPATIAL_DELTA_THETA_PREAGG_AUDIT"); audit && audit[0]) {
    std::vector<double> host(count);
    check_compact(cudaMemcpyAsync(host.data(), mean, count*sizeof(double), cudaMemcpyDeviceToHost,
                                  executor->contexts_[0]->stream()), "cudaMemcpyAsync(mean audit)");
    check_compact(cudaStreamSynchronize(executor->contexts_[0]->stream()),
                  "cudaStreamSynchronize(mean audit)");
    double sum=0., weighted=0.;
    for (size_t i=0;i<host.size();++i) { sum += host[i]; weighted += double(i+1)*host[i]; }
    std::ofstream output(audit, std::ios::app);
    if (!output) throw std::runtime_error("cannot open FEDSPATIAL_DELTA_THETA_PREAGG_AUDIT");
    output << std::setprecision(17) << "mu_sigma_mean device=" << central_device
           << " sum=" << sum << " weighted=" << weighted << '\n';
    output.flush();
  }
  *device_mean_mu = mean;
  *device_mean_sigma = mean + m;
}

// Move central global mu/Sigma, not raw inputs, to every logical worker.
void broadcast_mu_sigma(NWorkerPSynchronousExecutor* executor, const double* mu,
                        const double* sigma, int m) {
  const int central=executor->contexts_[0]->device_id();
  const size_t mu_bytes=size_t(m)*sizeof(double), sigma_bytes=size_t(m)*m*sizeof(double);
  const size_t bytes=mu_bytes+sigma_bytes;
  if(executor->pinned_compact_capacity_<bytes) { if(executor->pinned_compact_) check_round_cuda(cudaFreeHost(executor->pinned_compact_)); check_round_cuda(cudaMallocHost(&executor->pinned_compact_,bytes)); executor->pinned_compact_capacity_=bytes; }
  for(int worker=0;worker<executor->worker_count();++worker) {
    GpuContext& target_context=*executor->contexts_[worker];
    auto& target=const_cast<RegionGpuState&>(target_context.region()); const int destination=target_context.device_id();
    if(destination==central) { check_round_cuda(cudaSetDevice(central)); check_round_cuda(cudaMemcpyAsync(target.parameter_mu,mu,mu_bytes,cudaMemcpyDeviceToDevice,target_context.stream())); check_round_cuda(cudaMemcpyAsync(target.parameter_sigma,sigma,sigma_bytes,cudaMemcpyDeviceToDevice,target_context.stream())); }
    else { int peer=0; check_round_cuda(cudaDeviceCanAccessPeer(&peer,destination,central));
      if(peer) { check_round_cuda(cudaSetDevice(destination)); const cudaError_t enabled=cudaDeviceEnablePeerAccess(central,0); if(enabled!=cudaSuccess&&enabled!=cudaErrorPeerAccessAlreadyEnabled)check_round_cuda(enabled); if(enabled==cudaErrorPeerAccessAlreadyEnabled)(void)cudaGetLastError(); check_round_cuda(cudaMemcpyPeerAsync(target.parameter_mu,destination,mu,central,mu_bytes,target_context.stream())); check_round_cuda(cudaMemcpyPeerAsync(target.parameter_sigma,destination,sigma,central,sigma_bytes,target_context.stream())); executor->peer_summary_bytes_+=static_cast<long long>(bytes); }
      else { auto* pinned=static_cast<double*>(executor->pinned_compact_); check_round_cuda(cudaSetDevice(central)); check_round_cuda(cudaMemcpyAsync(pinned,mu,mu_bytes,cudaMemcpyDeviceToHost,executor->contexts_[0]->stream())); check_round_cuda(cudaMemcpyAsync(pinned+ m,sigma,sigma_bytes,cudaMemcpyDeviceToHost,executor->contexts_[0]->stream())); check_round_cuda(cudaStreamSynchronize(executor->contexts_[0]->stream())); check_round_cuda(cudaSetDevice(destination)); check_round_cuda(cudaMemcpyAsync(target.parameter_mu,pinned,mu_bytes,cudaMemcpyHostToDevice,target_context.stream())); check_round_cuda(cudaMemcpyAsync(target.parameter_sigma,pinned+m,sigma_bytes,cudaMemcpyHostToDevice,target_context.stream())); executor->pinned_summary_bytes_+=static_cast<long long>(bytes); }
    }
  }
  for(const auto& context:executor->contexts_){check_round_cuda(cudaSetDevice(context->device_id()));check_round_cuda(cudaStreamSynchronize(context->stream()));}
}

// Gather the local 13-double objective/gradient/Hessian records into fixed
// logical-worker slots on worker 0.  This intentionally mirrors the already
// audited mu/sigma and gamma compact-link policy.
void gather_compact_delta_theta(NWorkerPSynchronousExecutor* executor,
                                const std::vector<DeviceDeltaThetaSummary>& local,
                                double** total) {
  constexpr size_t count = 13, bytes = count * sizeof(double);
  const int central = executor->contexts_[0]->device_id();
  const int workers = static_cast<int>(local.size());
  check_round_cuda(cudaSetDevice(central));
  if (executor->compact_aggregation_capacity_ < count * size_t(workers + 1)) {
    if (executor->compact_aggregation_) check_round_cuda(cudaFree(executor->compact_aggregation_));
    check_round_cuda(cudaMalloc(&executor->compact_aggregation_, count * size_t(workers + 1) * sizeof(double)));
    executor->compact_aggregation_capacity_ = count * size_t(workers + 1);
  }
  if (executor->pinned_compact_capacity_ < bytes) {
    if (executor->pinned_compact_) check_round_cuda(cudaFreeHost(executor->pinned_compact_));
    check_round_cuda(cudaMallocHost(&executor->pinned_compact_, bytes));
    executor->pinned_compact_capacity_ = bytes;
  }
  for (int worker = 0; worker < workers; ++worker) {
    double* dst = executor->compact_aggregation_ + count * worker;
    const int source = executor->contexts_[worker]->device_id();
    // Local evaluation used a distinct worker stream.  Establish an explicit
    // producer/consumer boundary before a cross-context copy; without it a
    // peer DMA can observe an unfinished 13-double record.
    check_round_cuda(cudaSetDevice(source));
    check_round_cuda(cudaStreamSynchronize(executor->contexts_[worker]->stream()));
    check_round_cuda(cudaSetDevice(central));
    if (source == central) {
      check_round_cuda(cudaMemcpyAsync(dst, local[worker].values, bytes, cudaMemcpyDeviceToDevice,
                                       executor->contexts_[0]->stream()));
    } else {
      int peer = 0; check_round_cuda(cudaDeviceCanAccessPeer(&peer, central, source));
      if (peer) {
        const cudaError_t enabled = cudaDeviceEnablePeerAccess(source, 0);
        if (enabled != cudaSuccess && enabled != cudaErrorPeerAccessAlreadyEnabled) check_round_cuda(enabled);
        if (enabled == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
        check_round_cuda(cudaMemcpyPeerAsync(dst, central, local[worker].values, source, bytes,
                                              executor->contexts_[0]->stream()));
        executor->peer_summary_bytes_ += static_cast<long long>(bytes);
      } else {
        auto* pinned = static_cast<double*>(executor->pinned_compact_);
        check_round_cuda(cudaSetDevice(source));
        check_round_cuda(cudaMemcpyAsync(pinned, local[worker].values, bytes, cudaMemcpyDeviceToHost,
                                         executor->contexts_[worker]->stream()));
        check_round_cuda(cudaStreamSynchronize(executor->contexts_[worker]->stream()));
        check_round_cuda(cudaSetDevice(central));
        check_round_cuda(cudaMemcpyAsync(dst, pinned, bytes, cudaMemcpyHostToDevice,
                                         executor->contexts_[0]->stream()));
        executor->pinned_summary_bytes_ += static_cast<long long>(bytes);
      }
    }
  }
  double* reduced = executor->compact_aggregation_ + count * workers;
  reduce_compact_delta_theta<<<1, 1, 0, executor->contexts_[0]->stream()>>>(
      executor->compact_aggregation_, reduced, workers);
  check_round_cuda(cudaGetLastError());
  *total = reduced;
}

// Broadcast only the three updated scalar parameters.  The CPU only starts
// the copies; worker state remains device-resident and every relay is counted.
void broadcast_delta_theta(NWorkerPSynchronousExecutor* executor,
                           const double* updated) {
  constexpr size_t bytes=3*sizeof(double);
  const int central=executor->contexts_[0]->device_id();
  for (int worker=0; worker<executor->worker_count(); ++worker) {
    GpuContext& target_context=*executor->contexts_[worker];
    auto& target=const_cast<RegionGpuState&>(target_context.region());
    const int destination=target_context.device_id();
    if (destination==central) {
      check_round_cuda(cudaSetDevice(central));
      check_round_cuda(cudaMemcpyAsync(target.parameter_theta,updated,bytes,
                                       cudaMemcpyDeviceToDevice,target_context.stream()));
    } else {
      int peer=0; check_round_cuda(cudaDeviceCanAccessPeer(&peer,destination,central));
      if(peer) {
        check_round_cuda(cudaSetDevice(destination));
        const cudaError_t enabled=cudaDeviceEnablePeerAccess(central,0);
        if(enabled!=cudaSuccess&&enabled!=cudaErrorPeerAccessAlreadyEnabled) check_round_cuda(enabled);
        if(enabled==cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
        check_round_cuda(cudaMemcpyPeerAsync(target.parameter_theta,destination,updated,central,bytes,target_context.stream()));
        executor->peer_summary_bytes_+=bytes;
      } else {
        auto* pinned=static_cast<double*>(executor->pinned_compact_);
        check_round_cuda(cudaSetDevice(central));
        check_round_cuda(cudaMemcpyAsync(pinned,updated,bytes,cudaMemcpyDeviceToHost,executor->contexts_[0]->stream()));
        check_round_cuda(cudaStreamSynchronize(executor->contexts_[0]->stream()));
        check_round_cuda(cudaSetDevice(destination));
        check_round_cuda(cudaMemcpyAsync(target.parameter_theta,pinned,bytes,cudaMemcpyHostToDevice,target_context.stream()));
        executor->pinned_summary_bytes_+=bytes;
      }
    }
  }
  for(const auto& context:executor->contexts_) { check_round_cuda(cudaSetDevice(context->device_id())); check_round_cuda(cudaStreamSynchronize(context->stream())); }
}
std::string exception_details(const std::exception& error) {
  std::ostringstream out;
  out << "type=" << typeid(error).name() << " what=" << error.what();
  if (const auto* system = dynamic_cast<const std::system_error*>(&error))
    out << " code=" << system->code().value()
        << " category=" << system->code().category().name()
        << " message=" << system->code().message();
  return out.str();
}
[[noreturn]] void throw_stage_exception(const char* stage, int worker_id,
                                        int device_id, int round,
                                        const std::exception& error) {
  throw std::runtime_error(std::string("P worker failure stage=") + stage +
      " worker=" + std::to_string(worker_id) + " device=" +
      std::to_string(device_id) + " round=" + std::to_string(round) +
      " " + exception_details(error));
}
void add(Matrix& target, const Matrix& source, double scale = 1.0) {
  for (int i = 0; i < target.rows * target.cols; ++i) target.a[i] += scale * source.a[i];
}
double trace(const Matrix& matrix) { double result = 0.0; for (int i = 0; i < matrix.rows; ++i) result += matrix(i, i); return result; }
double inner(const Matrix& left, const Matrix& right) { double result = 0.0; for (int i = 0; i < left.rows * left.cols; ++i) result += left.a[i] * right.a[i]; return result; }
struct Eigen3 { Matrix vectors; std::array<double, 3> values; };
Eigen3 symmetric_eigen3(Matrix matrix) {
  Matrix vectors = identity(3);
  for (int iteration = 0; iteration < 64; ++iteration) {
    int p = 0, q = 1; double maximum = std::abs(matrix(0, 1));
    for (int i = 0; i < 3; ++i) for (int j = i + 1; j < 3; ++j)
      if (std::abs(matrix(i, j)) > maximum) { maximum = std::abs(matrix(i, j)); p = i; q = j; }
    if (maximum < 1e-15) break;
    const double angle = .5 * std::atan2(2. * matrix(p, q), matrix(q, q) - matrix(p, p));
    const double c = std::cos(angle), s = std::sin(angle);
    for (int k = 0; k < 3; ++k) { const double pk = matrix(p, k), qk = matrix(q, k); matrix(p, k) = c * pk - s * qk; matrix(q, k) = s * pk + c * qk; }
    for (int k = 0; k < 3; ++k) { const double kp = matrix(k, p), kq = matrix(k, q); matrix(k, p) = c * kp - s * kq; matrix(k, q) = s * kp + c * kq; const double vp = vectors(k, p), vq = vectors(k, q); vectors(k, p) = c * vp - s * vq; vectors(k, q) = s * vp + c * vq; }
  }
  std::array<int, 3> order{0, 1, 2}; std::array<double, 3> raw{matrix(0,0), matrix(1,1), matrix(2,2)};
  std::sort(order.begin(), order.end(), [&](int i, int j) { return raw[i] < raw[j]; });
  Matrix sorted_vectors(3, 3); std::array<double, 3> sorted_values{};
  for (int column = 0; column < 3; ++column) { sorted_values[column] = raw[order[column]]; for (int row = 0; row < 3; ++row) sorted_vectors(row, column) = vectors(row, order[column]); }
  return {sorted_vectors, sorted_values};
}
void global_k_correction(GpuContext& context, const Matrix& knots, const GlobalParameters& p,
                         Matrix* gradient, Matrix* hessian) {
  const auto d = matern_with_derivatives(context, knots, knots, p.nu2, p.alpha, p.range);
  Matrix first[3] = {Matrix(knots.rows, knots.rows), d.alpha, d.range};
  Matrix second[3][3] = {{Matrix(knots.rows, knots.rows), Matrix(knots.rows, knots.rows), Matrix(knots.rows, knots.rows)},
                          {Matrix(knots.rows, knots.rows), d.aa, d.ar},
                          {Matrix(knots.rows, knots.rows), d.ra, d.rr}};
  const Matrix inverse = solve_spd(d.value, identity(knots.rows));
  const Matrix inverse_mu = matmul(inverse, p.mu);
  for (int i = 1; i < 3; ++i) {
    (*gradient)(i, 0) = trace(matmul(inverse, first[i])) - inner(inverse_mu, matmul(first[i], inverse_mu)) - trace(matmul(matmul(matmul(inverse, first[i]), inverse), p.sigma));
    for (int j = 1; j < 3; ++j) {
      (*hessian)(i, j) = trace(matmul(inverse, second[i][j])) - trace(matmul(matmul(matmul(inverse, first[j]), inverse), first[i])) - inner(inverse_mu, matmul(second[i][j], inverse_mu)) + inner(inverse_mu, matmul(matmul(first[j], inverse), matmul(first[i], inverse_mu))) + inner(inverse_mu, matmul(matmul(first[i], inverse), matmul(first[j], inverse_mu))) + trace(matmul(matmul(matmul(matmul(inverse, first[j]), inverse), first[i]), matmul(inverse, p.sigma))) + trace(matmul(matmul(matmul(matmul(inverse, first[i]), inverse), first[j]), matmul(inverse, p.sigma))) - trace(matmul(matmul(matmul(inverse, second[i][j]), inverse), p.sigma));
    }
  }
}
}  // namespace

double compact_reduction_m2000_max_abs_error(GpuContext& context) {
  constexpr int m=2000, workers=2, threads=256;
  const int count=m+m*m;
  const size_t values=size_t(count)*(workers+1);
  double* device{};
  try {
    check_round_cuda(cudaSetDevice(context.device_id()));
    check_round_cuda(cudaMalloc(&device,values*sizeof(double)));
    initialize_compact_slots<<<(count+threads-1)/threads,threads,0,context.stream()>>>(device,count,workers);
    check_round_cuda(cudaGetLastError());
    double* mean=device+size_t(workers)*count;
    reduce_compact_mu_sigma<<<(count+threads-1)/threads,threads,0,context.stream()>>>(device,mean,workers,m,2.);
    check_round_cuda(cudaGetLastError());
    std::vector<double> result(count);
    check_round_cuda(cudaMemcpyAsync(result.data(),mean,size_t(count)*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));
    check_round_cuda(cudaStreamSynchronize(context.stream()));
    double error=0.;
    for (int index=0; index<count; ++index) {
      const double value=1.5+.125*double(index%13);
      const double expected=(index<m?-value:value)/2.;
      error=std::max(error,std::abs(result[index]-expected));
    }
    reduce_compact_gamma<<<(count+threads-1)/threads,threads,0,context.stream()>>>(device,mean,workers,count);
    check_round_cuda(cudaGetLastError());
    check_round_cuda(cudaMemcpyAsync(result.data(),mean,size_t(count)*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));
    check_round_cuda(cudaStreamSynchronize(context.stream()));
    for (int index=0; index<count; ++index)
      error=std::max(error,std::abs(result[index]-(1.5+.125*double(index%13))));
    check_round_cuda(cudaFree(device));
    return error;
  } catch (...) { if (device) cudaFree(device); throw; }
}

void apply_device_global_k_correction(GpuContext& context, double* total) {
  if (!total) throw std::runtime_error("null global-K summary");
  check_round_cuda(cudaSetDevice(context.device_id()));
  auto& r=context.region_;
  if (!r.initialized || !r.parameter_mu || !r.parameter_sigma)
    throw std::runtime_error("global-K correction requires the active device P state");
  const int n=r.m; const size_t nn=size_t(n)*n;
  if (!r.k_alpha) {
    for (double** pointer : {&r.k_alpha,&r.k_range,&r.k_alpha_alpha,
                             &r.k_alpha_range,&r.k_range_range})
      check_round_cuda(cudaMalloc(pointer,nn*sizeof(double)));
    ++r.global_k_correction_allocation_count;
  }
  global_k_derivatives<<<(n*n+255)/256,256,0,context.stream_>>>(
      r.knots,n,r.parameter_nu2,r.parameter_alpha_value,r.parameter_range_value,
      r.k_alpha,r.k_range,r.k_alpha_alpha,r.k_alpha_range,r.k_range_range);
  check_round_cuda(cudaGetLastError());
  // The streamed local evaluator owns no dense reduction arena.  The global
  // K correction needs only ten ordered matrix slots (not the retired 25
  // local-P slots); its vector is borrowed from the streamed fixed state.
  if(!r.delta_theta_reduction_scratch) {
    check_round_cuda(cudaMalloc(&r.delta_theta_reduction_scratch,10*nn*sizeof(double)));
    r.delta_theta_reduction_scratch_capacity=10*nn;
    ++r.delta_theta_reduction_allocation_count;
  }
  double* s=r.delta_theta_reduction_scratch;
  double* l=s; double* w=s+nn; double* p=s+2*nn;
  // Slots 2..7 hold the six product results and slots 8/9 are GEMM
  // temporaries.  The streamed inverse-column buffer is otherwise idle here.
  double* u=r.delta_theta_r;
  double* k_factor = r.chol_k ? r.chol_k : l;
  inverse_spd_parallel(context,r.k,k_factor,w,n,r.chol_k != nullptr && !use_scalar_global_inverse_reference());
  const double one=1.,zero=0.;
  check_round_blas_enqueue(context,cublasDgemv(context.cublas(),CUBLAS_OP_N,n,n,&one,w,n,
                                                r.parameter_mu,1,&zero,u,1),"DGEMV-global-K-u");
  const double* first[3]={nullptr,r.k_alpha,r.k_range};
  const double* second[9]={nullptr,nullptr,nullptr,nullptr,r.k_alpha_alpha,
                           r.k_alpha_range,nullptr,r.k_alpha_range,r.k_range_range};
  // Gradient: W Ki, then (W Ki) W, retaining legacy term order.
  for(int i=1;i<3;++i) {
    global_k_product(context,w,first[i],p,n);
    global_k_product(context,p,w,p+nn,n);
    global_k_gradient_add<<<1,1,0,context.stream_>>>(p,first[i],p+nn,
        r.parameter_sigma,u,total,1+i,n);
    check_round_cuda(cudaGetLastError());
  }
  // Evaluate only the upper triangle and mirror the mixed entry exactly.
  for(int i=1;i<3;++i) for(int j=i;j<3;++j) {
    const double* ki=first[i]; const double* kj=first[j]; const double* kij=second[3*i+j];
    // p[0]=W Kj W Ki; p[1]=W Kij W; p[2]=Kj W Ki;
    // p[3]=Ki W Kj; p[4]=W Kj W Ki W; p[5]=W Ki W Kj W.
    double* t0=p+6*nn; double* t1=p+7*nn;
    global_k_product(context,w,kj,t0,n); global_k_product(context,t0,w,t1,n); global_k_product(context,t1,ki,p+0*nn,n);
    global_k_product(context,w,kij,t0,n); global_k_product(context,t0,w,p+1*nn,n);
    global_k_product(context,kj,w,t0,n); global_k_product(context,t0,ki,p+2*nn,n);
    global_k_product(context,ki,w,t0,n); global_k_product(context,t0,kj,p+3*nn,n);
    global_k_product(context,w,kj,t0,n); global_k_product(context,t0,w,t1,n); global_k_product(context,t1,ki,t0,n); global_k_product(context,t0,w,p+4*nn,n);
    global_k_product(context,w,ki,t0,n); global_k_product(context,t0,w,t1,n); global_k_product(context,t1,kj,t0,n); global_k_product(context,t0,w,p+5*nn,n);
    global_k_hessian_add<<<1,1,0,context.stream_>>>(w,kij,p+0*nn,p+1*nn,p+2*nn,p+3*nn,p+4*nn,p+5*nn,r.parameter_sigma,u,total,4+3*i+j,n);
    check_round_cuda(cudaGetLastError());
  }
  global_k_mirror<<<1,1,0,context.stream_>>>(total);
  check_round_cuda(cudaGetLastError());
}

void compute_legacy_global_k_correction_for_test(
    GpuContext& context, const Matrix& knots, const GlobalParameters& parameters,
    Matrix* gradient, Matrix* hessian) {
  if (!gradient || !hessian) throw std::runtime_error("null legacy correction output");
  *gradient = Matrix(3, 1);
  *hessian = Matrix(3, 3);
  global_k_correction(context, knots, parameters, gradient, hessian);
}

Matrix deterministic_worker_order_mean(const std::vector<Matrix>& values) {
  if (values.empty()) throw std::runtime_error("empty worker reduction");
  Matrix result(values.front().rows, values.front().cols);
  for (const Matrix& value : values) {
    if (value.rows != result.rows || value.cols != result.cols)
      throw std::runtime_error("worker reduction shape mismatch");
    add(result, value);
  }
  for (double& value : result.a) value /= static_cast<double>(values.size());
  return result;
}

PSynchronousRoundResult run_p_synchronous_round(GpuContext& context, const std::vector<WorkerInput>& workers,
                                                  const Matrix& knots, const GlobalParameters& parameters) {
  PProfileScope profile_scope(context.profile(), "run_p_synchronous_round");
  if (workers.empty()) throw std::runtime_error("P round requires workers");
  const int j_count = static_cast<int>(workers.size());
  const int m = knots.rows, q = parameters.gamma.rows;
  Matrix mean_mu(m, 1), mean_sigma(m, m);
  for (const auto& worker : workers) { const auto local = compute_mu_sigma_local(context, worker, knots, parameters); add(mean_mu, local.summary.mu); add(mean_sigma, local.summary.sigma); }
  for (double& value : mean_mu.a) value /= j_count; for (double& value : mean_sigma.a) value /= j_count;
  const Matrix k = matern_with_derivatives(context, knots, knots, parameters.nu2, parameters.alpha, parameters.range).value;
  Matrix precision = solve_spd(k, identity(m)); add(precision, mean_sigma, j_count * parameters.delta);
  GlobalParameters state = parameters;
  state.sigma = solve_spd(precision, identity(m)); state.mu = matmul(state.sigma, mean_mu); for (double& value : state.mu.a) value *= -j_count * state.delta;
  Matrix mean_gamma(q, 1), mean_xx(q, q);
  for (const auto& worker : workers) { const auto local = compute_gamma_local(context, worker, knots, state); add(mean_gamma, local.gamma); add(mean_xx, local.xx); }
  for (double& value : mean_gamma.a) value /= j_count; for (double& value : mean_xx.a) value /= j_count;
  state.gamma = solve_spd(mean_xx, mean_gamma);
  Matrix local_gradient(3, 1), local_hessian(3, 3);
  for (const auto& worker : workers) { const auto local = compute_delta_theta_local(context, worker, knots, state); for (int i = 0; i < 3; ++i) { local_gradient(i,0) += local.gradient[i]; for (int j = 0; j < 3; ++j) local_hessian(i,j) += local.hessian[3*i+j]; } }
  for (double& value : local_gradient.a) value /= j_count; for (double& value : local_hessian.a) value /= j_count;
  Matrix correction_gradient(3, 1), correction_hessian(3, 3); global_k_correction(context, knots, state, &correction_gradient, &correction_hessian);
  for (double& value : local_gradient.a) value *= j_count; for (double& value : local_hessian.a) value *= j_count; add(local_gradient, correction_gradient); add(local_hessian, correction_hessian);
  const auto eigen = symmetric_eigen3(local_hessian); Matrix modified(3, 3);
  for (int k_index = 0; k_index < 3; ++k_index) { const double value = std::max(std::abs(eigen.values[k_index]), .01); for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column) modified(row,column) += eigen.vectors(row,k_index) * value * eigen.vectors(column,k_index); }
  const Matrix direction = solve_spd(modified, local_gradient);
  double step = 1.; int halvings = 0;
  double delta = state.delta - step * direction(0,0), alpha = state.alpha - step * direction(1,0), range = state.range - step * direction(2,0);
  while (delta <= 0. || alpha <= 0. || range <= 0.) { step *= .5; ++halvings; delta = state.delta - step * direction(0,0); alpha = state.alpha - step * direction(1,0); range = state.range - step * direction(2,0); }
  state.delta = delta; state.alpha = alpha; state.range = range;
  return {state, step, halvings};
}

NWorkerPSynchronousExecutor::NWorkerPSynchronousExecutor(std::vector<int> device_ids) {
  if (device_ids.empty()) throw std::runtime_error("N-worker executor requires a device ID per worker");
  contexts_.reserve(device_ids.size());
  for (const int device_id : device_ids)
    contexts_.push_back(std::make_unique<GpuContext>(device_id));
}

NWorkerPSynchronousExecutor::~NWorkerPSynchronousExecutor() {
  // Contexts must outlive the central allocations only for stream ordering;
  // the round method synchronizes before returning.
  if (!contexts_.empty()) cudaSetDevice(contexts_[0]->device_id());
  if (compact_aggregation_) cudaFree(compact_aggregation_);
  if (global_update_) cudaFree(global_update_);
  if (pinned_compact_) cudaFreeHost(pinned_compact_);
}

int NWorkerPSynchronousExecutor::worker_count() const {
  return static_cast<int>(contexts_.size());
}

PWorkerRoundCheckpoints NWorkerPSynchronousExecutor::run_round_with_checkpoints(
    const std::vector<WorkerInput>& workers, const Matrix& knots,
    const GlobalParameters& parameters) {
  if (workers.size() != contexts_.size())
    throw std::runtime_error("worker input count does not match executor contexts");
  const int round = this->round_sequence_++;
  // bind_region allocates and synchronizes each independent device stream.
  // It used to run in the caller thread in worker-ID order, serializing all
  // three initialization uploads before any local stage could start.
  stage_log(round, "bind_region", "BEGIN");
  std::vector<std::exception_ptr> bind_errors(contexts_.size());
  std::vector<std::thread> bind_threads;
  bind_threads.reserve(contexts_.size());
  for (int worker_id = 0; worker_id < worker_count(); ++worker_id) {
    bind_threads.emplace_back([&, worker_id] {
      const int device = contexts_[worker_id]->device_id();
      stage_log(round, "bind_region", "WORKER_BEGIN", worker_id, device);
      try {
        if (cudaSetDevice(device) != cudaSuccess)
          throw std::runtime_error("cudaSetDevice failed in bind_region");
        contexts_[worker_id]->bind_region(workers[worker_id], knots);
      } catch (...) {
        bind_errors[worker_id] = std::current_exception();
      }
      stage_log(round, "bind_region", "WORKER_END", worker_id, device);
    });
  }
  for (auto& thread : bind_threads) thread.join();
  for (int worker_id = 0; worker_id < worker_count(); ++worker_id) {
    if (!bind_errors[worker_id]) continue;
    try { std::rethrow_exception(bind_errors[worker_id]); }
    catch (const std::exception& error) {
      throw_stage_exception("bind_region", worker_id,
                            contexts_[worker_id]->device_id(), round, error);
    }
  }
  stage_log(round, "bind_region", "BARRIER_COMPLETE");
  std::vector<long long> raw_bytes_at_round_start;
  std::vector<long long> model_bytes_at_round_start;
  std::vector<long long> host_compute_bytes_at_round_start;
  raw_bytes_at_round_start.reserve(worker_count());
  model_bytes_at_round_start.reserve(worker_count());
  host_compute_bytes_at_round_start.reserve(worker_count());
  for (const auto& context : contexts_) {
    raw_bytes_at_round_start.push_back(context->raw_h2d_bytes());
    model_bytes_at_round_start.push_back(context->parameter_h2d_bytes());
    host_compute_bytes_at_round_start.push_back(context->host_compute_d2h_bytes());
  }
  const long long peer_bytes_at_round_start = peer_summary_bytes_;
  const long long pinned_bytes_at_round_start = pinned_summary_bytes_;
  // Each lambda writes a distinct result slot and owns a distinct CUDA stream,
  // cuBLAS handle, and cuSOLVER handle. Joining is the stage barrier.
  const auto round_start = std::chrono::steady_clock::now();
  PSynchronousRoundTiming timing;
  auto run_local = [&](const char* stage, auto&& operation, double* elapsed) {
    stage_log(round, stage, "BEGIN");
    const auto stage_start = std::chrono::steady_clock::now();
    using Result = decltype(operation(*contexts_[0], workers[0]));
    std::vector<Result> results(contexts_.size());
    std::vector<std::exception_ptr> errors(contexts_.size());
    std::vector<std::thread> threads;
    threads.reserve(contexts_.size());
    for (int worker_id = 0; worker_id < worker_count(); ++worker_id) {
      try {
        threads.emplace_back([&, worker_id] {
        try {
          if (cudaSetDevice(contexts_[worker_id]->device_id()) != cudaSuccess)
            throw std::runtime_error("cudaSetDevice failed in logical worker");
          stage_log(round, stage, "WORKER_BEGIN", worker_id,
                    contexts_[worker_id]->device_id());
          const bool cuda_timing = stage_cuda_timing_enabled();
          cudaEvent_t begin{}, finish{};
          if (cuda_timing) {
            check_round_cuda(cudaEventCreate(&begin));
            check_round_cuda(cudaEventCreate(&finish));
            check_round_cuda(cudaEventRecord(begin, contexts_[worker_id]->stream()));
          }
          const auto worker_start = std::chrono::steady_clock::now();
          results[worker_id] = operation(*contexts_[worker_id], workers[worker_id]);
          if (cuda_timing) {
            check_round_cuda(cudaEventRecord(finish, contexts_[worker_id]->stream()));
            check_round_cuda(cudaEventSynchronize(finish));
            float device_ms{};
            check_round_cuda(cudaEventElapsedTime(&device_ms, begin, finish));
            const double wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - worker_start).count();
            stage_timing_log(round, stage, worker_id, contexts_[worker_id]->device_id(),
                             wall_ms, device_ms);
            cudaEventDestroy(finish);
            cudaEventDestroy(begin);
          }
          stage_log(round, stage, "WORKER_END", worker_id,
                    contexts_[worker_id]->device_id());
        } catch (...) {
          errors[worker_id] = std::current_exception();
        }
        });
      } catch (const std::exception& error) {
        for (auto& thread : threads) if (thread.joinable()) thread.join();
        throw_stage_exception(stage, worker_id, contexts_[worker_id]->device_id(),
                              round, error);
      }
    }
    // Explicit joins make error propagation and stage boundaries unambiguous.
    for (int worker_id = 0; worker_id < worker_count(); ++worker_id) {
      try { threads[worker_id].join(); }
      catch (const std::exception& error) {
        throw_stage_exception(stage, worker_id, contexts_[worker_id]->device_id(),
                              round, error);
      }
    }
    for (int worker_id = 0; worker_id < worker_count(); ++worker_id) if (errors[worker_id]) {
      try { std::rethrow_exception(errors[worker_id]); }
      catch (const std::exception& error) {
        throw_stage_exception(stage, worker_id, contexts_[worker_id]->device_id(),
                              round, error);
      }
    }
    *elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();
    stage_log(round, stage, "BARRIER_COMPLETE");
    return results;
  };

  const int worker_count = this->worker_count();
  const int m = knots.rows;
  const auto mu_sigma = run_local("mu_sigma", [&](GpuContext& context, const WorkerInput& worker) {
    // Keep both compact sufficient statistics on the worker GPU.  In
    // particular do not call the diagnostic model-state downloader here.
    try { return build_device_mu_sigma_local(context, worker, knots, parameters); }
    catch (const std::exception& error) { throw std::runtime_error(std::string("mu/sigma local: ") + error.what()); }
  }, &timing.mu_sigma_local_seconds);
  auto aggregation_start = std::chrono::steady_clock::now();
  stage_log(round, "mu_sigma_aggregate", "BEGIN");
  double *mean_mu{}, *mean_sigma{};
  gather_compact_mu_sigma(this, mu_sigma, m, parameters.delta,
                          &mean_mu, &mean_sigma);
  timing.aggregation_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - aggregation_start).count();
  stage_log(round, "mu_sigma_aggregate", "COMPLETE");

  // The global state is produced once, then captured by value/read-only by
  // both logical workers in the next stage.
  auto update_start = std::chrono::steady_clock::now();
  stage_log(round, "mu_sigma_global_update", "BEGIN", 0, contexts_[0]->device_id());
  if (cudaSetDevice(contexts_[0]->device_id()) != cudaSuccess)
    throw std::runtime_error("cudaSetDevice failed before global mu/sigma covariance");
  const size_t nn=size_t(m)*m, required=3*nn+size_t(m);
  if(global_update_capacity_<required) {
    if(global_update_) check_round_cuda(cudaFree(global_update_));
    check_round_cuda(cudaMalloc(&global_update_,required*sizeof(double)));
    global_update_capacity_=required;
  }
  // K construction, K inverse solve, precision assembly and posterior solve
  // all execute in fixed FP64 order on the designated aggregation GPU.
  const auto k_state=compute_device_b_slice(*contexts_[0], parameters.nu2,
                                             parameters.alpha, parameters.range);
  double* precision=global_update_; double* precision_chol=precision+nn;
  double* sigma=precision_chol+nn; double* mu=sigma+nn;
  inverse_spd_parallel(*contexts_[0],k_state.k,const_cast<double*>(k_state.chol_k),precision,m,true);
  dump_device_audit(*contexts_[0], "global_k_inverse", precision, nn);
  add_precision<<<(m*m+255)/256,256,0,contexts_[0]->stream()>>>(precision,mean_sigma,worker_count*parameters.delta,m);
  check_round_cuda(cudaGetLastError());
  symmetrize_matrix<<<(m*m+255)/256,256,0,contexts_[0]->stream()>>>(precision,m);
  check_round_cuda(cudaGetLastError());
  dump_device_audit(*contexts_[0], "global_precision", precision, nn);
  inverse_spd_parallel(*contexts_[0],precision,precision_chol,sigma,m,false);
  dump_device_audit(*contexts_[0], "global_sigma", sigma, nn);
  symmetrize_matrix<<<(m*m+255)/256,256,0,contexts_[0]->stream()>>>(sigma,m);
  const double mu_scale=-worker_count*parameters.delta,zero=0.;
  check_round_blas_enqueue(*contexts_[0],cublasDgemv(contexts_[0]->cublas(),CUBLAS_OP_N,m,m,
      &mu_scale,sigma,m,mean_mu,1,&zero,mu,1),"DGEMV-global-mu");
  dump_device_audit(*contexts_[0], "global_mu", mu, size_t(m));
  auto& central_region=const_cast<RegionGpuState&>(contexts_[0]->region());
  check_round_cuda(cudaMemcpyAsync(central_region.parameter_mu,mu,size_t(m)*sizeof(double),cudaMemcpyDeviceToDevice,contexts_[0]->stream()));
  check_round_cuda(cudaMemcpyAsync(central_region.parameter_sigma,sigma,nn*sizeof(double),cudaMemcpyDeviceToDevice,contexts_[0]->stream()));
  broadcast_mu_sigma(this,mu,sigma,m);
  GlobalParameters state = parameters;
  // Compatibility-only checkpoint download; neither array is an input to
  // the active server arithmetic above.
  check_round_cuda(cudaSetDevice(contexts_[0]->device_id()));
  check_round_cuda(cudaMemcpyAsync(state.mu.a.data(),mu,size_t(m)*sizeof(double),cudaMemcpyDeviceToHost,contexts_[0]->stream()));
  check_round_cuda(cudaMemcpyAsync(state.sigma.a.data(),sigma,nn*sizeof(double),cudaMemcpyDeviceToHost,contexts_[0]->stream()));
  timing.model_d2h_bytes += static_cast<long long>((size_t(m) + nn) * sizeof(double));
  check_round_cuda(cudaStreamSynchronize(contexts_[0]->stream()));
  timing.global_update_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
  stage_log(round, "mu_sigma_global_update", "COMPLETE", 0, contexts_[0]->device_id());
  if (const char* audit = std::getenv("FEDSPATIAL_DELTA_THETA_PREAGG_AUDIT"); audit && audit[0]) {
    double mu_sum=0., mu_weighted=0., sigma_sum=0., sigma_weighted=0.;
    for (size_t i=0;i<state.mu.a.size();++i) { mu_sum += state.mu.a[i]; mu_weighted += double(i+1)*state.mu.a[i]; }
    for (size_t i=0;i<state.sigma.a.size();++i) { sigma_sum += state.sigma.a[i]; sigma_weighted += double(i+1)*state.sigma.a[i]; }
    std::ofstream output(audit, std::ios::app);
    if (!output) throw std::runtime_error("cannot open FEDSPATIAL_DELTA_THETA_PREAGG_AUDIT");
    output << std::setprecision(17) << "round=" << round << " after_mu_sigma"
           << " mu_sum=" << mu_sum << " mu_weighted=" << mu_weighted
           << " sigma_sum=" << sigma_sum << " sigma_weighted=" << sigma_weighted << '\n';
    output.flush();
  }
  const GlobalParameters after_mu_sigma = state;

  const auto gamma = run_local("gamma", [&](GpuContext& context, const WorkerInput& worker) {
    try { return compute_device_gamma_local(context, worker, knots, state); }
    catch (const std::exception& error) { throw std::runtime_error(std::string("gamma local: ") + error.what()); }
  }, &timing.gamma_local_seconds);
  aggregation_start = std::chrono::steady_clock::now();
  stage_log(round, "gamma_aggregate", "BEGIN");
  double *mean_gamma{}, *mean_xx{};
  try { gather_compact_gamma(this, gamma, state.gamma.rows, &mean_gamma, &mean_xx); }
  catch (const std::exception& error) { throw std::runtime_error(std::string("gamma compact gather: ") + error.what()); }
  timing.aggregation_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - aggregation_start).count();
  stage_log(round, "gamma_aggregate", "COMPLETE");
  update_start = std::chrono::steady_clock::now();
  stage_log(round, "gamma_global_update", "BEGIN", 0, contexts_[0]->device_id());
  try { solve_and_broadcast_gamma(this, mean_gamma, mean_xx, state.gamma.rows); }
  catch (const std::exception& error) { throw std::runtime_error(std::string("gamma GPU solve/broadcast: ") + error.what()); }
  // This compact diagnostic download is deliberately not used for arithmetic;
  // state.gamma remains host-visible only for checkpoint compatibility.
  try {
    check_round_cuda(cudaSetDevice(contexts_[0]->device_id()));
    check_round_cuda(cudaMemcpyAsync(state.gamma.a.data(), mean_gamma,
                                     size_t(state.gamma.rows) * sizeof(double), cudaMemcpyDeviceToHost,
                                     contexts_[0]->stream()));
    timing.model_d2h_bytes += static_cast<long long>(size_t(state.gamma.rows) * sizeof(double));
    check_round_cuda(cudaStreamSynchronize(contexts_[0]->stream()));
  } catch (const std::exception& error) { throw std::runtime_error(std::string("gamma checkpoint download: ") + error.what()); }
  timing.global_update_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
  stage_log(round, "gamma_global_update", "COMPLETE", 0, contexts_[0]->device_id());
  const GlobalParameters after_gamma = state;
  dump_after_gamma_audit(state, round);

  const auto theta = run_local("delta_theta", [&](GpuContext& context, const WorkerInput& worker) {
    // Do not materialize local objective/gradient/Hessian records on the
    // host.  The following gather owns the sole transport boundary.
    try { return compute_device_delta_theta_summary(context, worker, knots, state); }
    catch (const std::exception& error) { throw std::runtime_error(std::string("delta/theta local: ") + error.what()); }
  }, &timing.delta_theta_local_seconds);
  dump_delta_theta_preaggregation_audit(theta, contexts_, round);
  aggregation_start = std::chrono::steady_clock::now();
  stage_log(round, "delta_theta_aggregate", "BEGIN");
  std::array<double, 5> delta_theta_checkpoint{};
  try {
    double* total{}; gather_compact_delta_theta(this, theta, &total);
    // The designated aggregation GPU owns K, its derivative bundle, current
    // mu/Sigma, and the complete 3x3 Newton update.  The download below is a
    // compact checkpoint, never an arithmetic input.
    // The local 13-double producer is complete; retire its three n-by-m pair
    // buffers before the designated central m-by-m correction begins.
    release_device_delta_theta_pair_workspace(*contexts_[0]);
    apply_device_global_k_correction(*contexts_[0], total);
    dump_device_audit(*contexts_[0], "global_delta_theta_corrected", total, 13);
    if (global_update_capacity_ < 5) {
      check_round_cuda(cudaSetDevice(contexts_[0]->device_id()));
      if (global_update_) check_round_cuda(cudaFree(global_update_));
      check_round_cuda(cudaMalloc(&global_update_, 5*sizeof(double)));
      global_update_capacity_=5;
    }
    global_newton_update<<<1,1,0,contexts_[0]->stream()>>>(total,state.delta,state.alpha,state.range,global_update_);
    check_round_cuda(cudaGetLastError());
    dump_device_audit(*contexts_[0], "global_newton_update", global_update_, 5);
    broadcast_delta_theta(this, global_update_);
    check_round_cuda(cudaSetDevice(contexts_[0]->device_id()));
    check_round_cuda(cudaMemcpyAsync(delta_theta_checkpoint.data(),global_update_,
                                     delta_theta_checkpoint.size()*sizeof(double),cudaMemcpyDeviceToHost,
                                     contexts_[0]->stream()));
    timing.model_d2h_bytes += static_cast<long long>(delta_theta_checkpoint.size()*sizeof(double));
    check_round_cuda(cudaStreamSynchronize(contexts_[0]->stream()));
  } catch (const std::exception& error) { throw std::runtime_error(std::string("delta/theta compact aggregation: ") + error.what()); }
  timing.aggregation_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - aggregation_start).count();
  stage_log(round, "delta_theta_aggregate", "COMPLETE");
  update_start = std::chrono::steady_clock::now();
  state.delta=delta_theta_checkpoint[0]; state.alpha=delta_theta_checkpoint[1];
  state.range=delta_theta_checkpoint[2];
  const double step=delta_theta_checkpoint[3]; const int halvings=static_cast<int>(delta_theta_checkpoint[4]);
  timing.global_update_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
  for (int worker_id = 0; worker_id < worker_count; ++worker_id)
    timing.raw_data_stationary = timing.raw_data_stationary &&
        contexts_[worker_id]->raw_h2d_bytes() == raw_bytes_at_round_start[worker_id];
  for (int worker_id = 0; worker_id < worker_count; ++worker_id) {
    timing.raw_h2d_bytes += contexts_[worker_id]->raw_h2d_bytes() - raw_bytes_at_round_start[worker_id];
    timing.model_h2d_bytes += contexts_[worker_id]->parameter_h2d_bytes() - model_bytes_at_round_start[worker_id];
    timing.host_compute_d2h_bytes += contexts_[worker_id]->host_compute_d2h_bytes() - host_compute_bytes_at_round_start[worker_id];
    timing.model_data_stationary = timing.model_data_stationary &&
        contexts_[worker_id]->parameter_h2d_bytes() == model_bytes_at_round_start[worker_id];
  }
  if (!timing.raw_data_stationary)
    throw std::runtime_error("raw worker data moved after round initialization");
  timing.peer_summary_bytes = peer_summary_bytes_ - peer_bytes_at_round_start;
  timing.pinned_summary_bytes = pinned_summary_bytes_ - pinned_bytes_at_round_start;
  timing.total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - round_start).count();
  stage_log(round, "round", "COMPLETE");
  return {after_mu_sigma, after_gamma, {state, step, halvings}, timing};
}

PSynchronousRoundResult NWorkerPSynchronousExecutor::run_round(
    const std::vector<WorkerInput>& workers, const Matrix& knots,
    const GlobalParameters& parameters) {
  return run_round_with_checkpoints(workers, knots, parameters).after_delta_theta;
}

TwoWorkerPSynchronousExecutor::TwoWorkerPSynchronousExecutor(int device_id_worker0,
                                                               int device_id_worker1)
    : executor_(std::make_unique<NWorkerPSynchronousExecutor>(
          std::vector<int>{device_id_worker0, device_id_worker1})) {}

PTwoWorkerRoundCheckpoints TwoWorkerPSynchronousExecutor::run_round_with_checkpoints(
    const std::array<WorkerInput, 2>& workers, const Matrix& knots,
    const GlobalParameters& parameters) {
  return executor_->run_round_with_checkpoints(
      std::vector<WorkerInput>{workers[0], workers[1]}, knots, parameters);
}

PSynchronousRoundResult TwoWorkerPSynchronousExecutor::run_round(
    const std::array<WorkerInput, 2>& workers, const Matrix& knots,
    const GlobalParameters& parameters) {
  return executor_->run_round(std::vector<WorkerInput>{workers[0], workers[1]},
                              knots, parameters);
}
}  // namespace fedspatial
