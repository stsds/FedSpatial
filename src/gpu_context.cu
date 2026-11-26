#include "fedspatial/gpu_context.hpp"
#include <cuda_fp16.h>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fedspatial {
namespace {
std::uint64_t fingerprint(const std::vector<double>& values) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (double value : values) { std::uint64_t bits{}; std::memcpy(&bits, &value, sizeof(bits)); hash = (hash ^ bits) * 1099511628211ULL; }
  return hash;
}
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void check(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS failure"); }
void check(cusolverStatus_t s) { if (s != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER failure"); }
__global__ void double_to_float(const double* x,float* y,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)y[i]=static_cast<float>(x[i]);}
__global__ void double_to_half(const double* x,__half* y,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)y[i]=__double2half(x[i]);}
__global__ void float_to_double(const float* x,double* y,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)y[i]=static_cast<double>(x[i]);}
void precision_capacity(void** p,size_t* capacity,size_t bytes){if(*capacity>=bytes)return;if(*p)check(cudaFree(*p));check(cudaMalloc(p,bytes));*capacity=bytes;}
void precision_input(const double*x,void*y,size_t n,ArithmeticPrecision mode,cudaStream_t stream){const int b=256,g=int((n+b-1)/b);if(mode==ArithmeticPrecision::fp32)double_to_float<<<g,b,0,stream>>>(x,static_cast<float*>(y),n);else double_to_half<<<g,b,0,stream>>>(x,static_cast<__half*>(y),n);check(cudaGetLastError());}
bool reduced_bkb(GpuContext&c,RegionGpuState&r,const double*k,const double*b,int n,int m){const auto mode=c.precision_policy().mode;if(mode==ArithmeticPrecision::fp64)return false;const size_t es=mode==ArithmeticPrecision::fp32?sizeof(float):sizeof(__half),mn=size_t(m)*n,mm=size_t(m)*m,nn=size_t(n)*n;precision_capacity(&r.precision_a,&r.precision_a_bytes,std::max(mm,mn)*es);precision_capacity(&r.precision_b,&r.precision_b_bytes,mn*es);precision_capacity(&r.precision_c,&r.precision_c_bytes,std::max(mn,nn)*sizeof(float));precision_input(k,r.precision_a,mm,mode,c.stream());precision_input(b,r.precision_b,mn,mode,c.stream());const cudaDataType_t t=mode==ArithmeticPrecision::fp32?CUDA_R_32F:CUDA_R_16F;const float one=1.f,zero=0.f;check(cublasGemmEx(c.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,m,&one,r.precision_a,t,m,r.precision_b,t,m,&zero,r.precision_c,CUDA_R_32F,m,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));float_to_double<<<int((mn+255)/256),256,0,c.stream()>>>(static_cast<float*>(r.precision_c),r.bk,mn);check(cudaGetLastError());precision_input(r.bk,r.precision_a,mn,mode,c.stream());check(cublasGemmEx(c.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,n,n,m,&one,r.precision_a,t,m,r.precision_b,t,m,&zero,r.precision_c,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));float_to_double<<<int((nn+255)/256),256,0,c.stream()>>>(static_cast<float*>(r.precision_c),r.m_matrix,nn);check(cudaGetLastError());r.precision_gemm_calls+=2;return true;}
__global__ void matern(const double* x, const double* y, double* out, int nx, int ny, int nu2, double a, double r) { int t=blockIdx.x*blockDim.x+threadIdx.x; if(t<nx*ny) { int i=t/ny,j=t%ny; double dx=x[2*i]-y[2*j],dy=x[2*i+1]-y[2*j+1],d=sqrt(dx*dx+dy*dy)/r,v; if(nu2==1)v=exp(-d); else if(nu2==3){double u=sqrt(3.)*d;v=(1.+u)*exp(-u);} else {double u=sqrt(5.)*d;v=(1.+u+u*u/3.)*exp(-u);} out[t]=a*v; } }
__global__ void make_m(const double* c, const double* bkb, double* m, int n, double d) { int t=blockIdx.x*blockDim.x+threadIdx.x; if(t<n*n) { int i=t/n,j=t%n; m[t]=d*(c[t]-bkb[t])+(i==j); } }
__global__ void residual(const double* z, const double* x, const double* g, double* e, int n, int q) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) {double xg=0;for(int k=0;k<q;k++)xg+=x[i+n*k]*g[k];e[i]=z[i]-xg;} }
__global__ void subtract_vec(double* x, const double* y, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]-=y[i]; }
__global__ void diag(const double* l, double* d, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=l[i*n+i]; }
// One owner preserves the legacy diagonal visitation and FP64 addition order.
__global__ void local_logdet(const double* chol, int n, double delta, double* values) {
  if (blockIdx.x || threadIdx.x) return;
  double value=0.;
  for (int i=0; i<n; ++i) value += 2.*log(chol[i*n+i]);
  values[1]=value-n*log(delta);
}
__global__ void finish_local_objective(double* values) {
  if (blockIdx.x || threadIdx.x) return;
  values[3]=values[1]+values[2]+values[3];
}
__global__ void set_identity(double* matrix, int n) { int t=blockIdx.x*blockDim.x+threadIdx.x; if(t<n*n) { int i=t/n,j=t%n; matrix[t]=(i==j)?1.:0.; } }
__global__ void set_zero(double* values, size_t count) { const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(i<count) values[i]=0.; }

// A P delta/theta derivative evaluation invokes several derivative helpers
// with the identical worker and fixed state.  The complete local result is
// immutable, so retaining the immediately preceding exact state avoids
// rebuilding/factoring the same K and M matrices.  This is thread-local: a
// production process owns one context/GPU and separate processes never share
// cached device-independent host results.
struct LocalResultCache {
  const GpuContext* context{};
  const Matrix* knots{};
  double delta{}, alpha{}, range{};
  // A caller may construct successive WorkerInput values in the same stack
  // slot (the Figure 1 local-Hessian gate does exactly this).  An address-only
  // key would then return a previous worker's immutable result.  Retain a
  // value snapshot for the cache predicate rather than using WorkerInput's
  // transient object address as its identity.
  Matrix worker_sites, worker_z, worker_x;
  Matrix gamma, mu, sigma;
  LocalComputationResult result;
  bool valid{};
};
thread_local LocalResultCache local_result_cache;

bool same_matrix(const Matrix& left, const Matrix& right) {
  return left.rows == right.rows && left.cols == right.cols && left.a == right.a;
}

bool cached_local_matches(const GpuContext& context, const WorkerInput& worker,
                          const Matrix& knots, const GlobalParameters& parameters) {
  const auto& cached = local_result_cache;
  return cached.valid && cached.context == &context && context.region().initialized &&
         cached.knots == &knots && cached.delta == parameters.delta &&
         cached.alpha == parameters.alpha && cached.range == parameters.range &&
         same_matrix(cached.worker_sites, worker.sites) &&
         same_matrix(cached.worker_z, worker.z) &&
         same_matrix(cached.worker_x, worker.x) &&
         same_matrix(cached.gamma, parameters.gamma) &&
         same_matrix(cached.mu, parameters.mu) && same_matrix(cached.sigma, parameters.sigma);
}

void cache_local_result(const GpuContext& context, const WorkerInput& worker,
                        const Matrix& knots, const GlobalParameters& parameters,
                        const LocalComputationResult& result) {
  auto& cached = local_result_cache;
  cached.context = &context; cached.knots = &knots;
  cached.delta = parameters.delta; cached.alpha = parameters.alpha; cached.range = parameters.range;
  cached.worker_sites = worker.sites; cached.worker_z = worker.z; cached.worker_x = worker.x;
  cached.gamma = parameters.gamma; cached.mu = parameters.mu; cached.sigma = parameters.sigma;
  cached.result = result; cached.valid = true;
}
}

GpuContext::GpuContext(int device_id) : device_id_(device_id) { check(cudaSetDevice(device_id_)); check(cudaStreamCreate(&stream_)); check(cublasCreate(&cublas_)); check(cusolverDnCreate(&cusolver_)); check(cublasSetStream(cublas_,stream_)); check(cusolverDnSetStream(cusolver_,stream_)); }
void GpuContext::set_local_solve_configuration(LocalSolveConfiguration configuration) {
  local_solve_configuration_ = configuration;
}
LocalSolveConfiguration GpuContext::local_solve_configuration() const {
  return local_solve_configuration_;
}
void GpuContext::set_precision_policy(PrecisionPolicy policy) { precision_policy_=policy; }
PrecisionPolicy GpuContext::precision_policy() const { return precision_policy_; }
bool gpu_precision_gemm(GpuContext& c,cublasOperation_t opa,cublasOperation_t opb,
                        int rows,int columns,int inner,const double* a,int lda,
                        const double* b,int ldb,double* out,int ldc){
  const auto mode=c.precision_policy().mode;
  if(mode==ArithmeticPrecision::fp64)return false;
  auto&r=const_cast<RegionGpuState&>(c.region());
  const size_t ae=size_t(lda)*(opa==CUBLAS_OP_N?inner:rows);
  const size_t be=size_t(ldb)*(opb==CUBLAS_OP_N?columns:inner);
  const size_t ce=size_t(ldc)*columns;
  const size_t es=mode==ArithmeticPrecision::fp32?sizeof(float):sizeof(__half);
  precision_capacity(&r.precision_a,&r.precision_a_bytes,ae*es);
  precision_capacity(&r.precision_b,&r.precision_b_bytes,be*es);
  precision_capacity(&r.precision_c,&r.precision_c_bytes,ce*sizeof(float));
  precision_input(a,r.precision_a,ae,mode,c.stream());
  precision_input(b,r.precision_b,be,mode,c.stream());
  const cudaDataType_t t=mode==ArithmeticPrecision::fp32?CUDA_R_32F:CUDA_R_16F;
  const float one=1.f,zero=0.f;
  check(cublasGemmEx(c.cublas(),opa,opb,rows,columns,inner,&one,
                     r.precision_a,t,lda,r.precision_b,t,ldb,&zero,
                     r.precision_c,CUDA_R_32F,ldc,CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  float_to_double<<<int((ce+255)/256),256,0,c.stream()>>>(
      static_cast<float*>(r.precision_c),out,ce);
  check(cudaGetLastError());
  ++r.precision_gemm_calls;
  return true;
}
GpuContext::~GpuContext() { cudaSetDevice(device_id_); if(region_.precision_a) cudaFree(region_.precision_a); if(region_.precision_b) cudaFree(region_.precision_b); if(region_.precision_c) cudaFree(region_.precision_c); if(region_.b_first) cudaFree(region_.b_first); if(region_.k_first) cudaFree(region_.k_first); for(double* p:{region_.sites,region_.z,region_.x_col_major,region_.knots,region_.cnk,region_.k,region_.chol_k,region_.b,region_.cjj,region_.cjj_alpha_alpha,region_.cjj_alpha_range,region_.cjj_range_range,region_.m_matrix,region_.chol_m,region_.bt_rinv_b,region_.producer_bt_minv_b,region_.bk,region_.residual_e,region_.minv_b,region_.minv_e,region_.bt_minv_e,region_.rhs,region_.chol_diag,region_.scalar_checkpoint,region_.parameter_gamma,region_.parameter_mu,region_.parameter_sigma,region_.parameter_delta,region_.parameter_theta,region_.gamma_x_solved,region_.gamma_local,region_.gamma_xx,region_.delta_theta_r,region_.delta_theta_u,region_.delta_theta_w,region_.delta_theta_q,region_.delta_theta_q_intermediate,region_.delta_theta_b_zero,region_.delta_theta_r_first,region_.delta_theta_r_second,region_.delta_theta_q_first,region_.delta_theta_q_second,region_.delta_theta_q_scratch0,region_.delta_theta_q_scratch1,region_.delta_theta_reduction_scratch,region_.delta_theta_summary,region_.delta_theta_stream_partials,region_.mixed_action_tile_in,region_.mixed_action_tile_out,region_.mixed_action_tile_m0,region_.mixed_action_tile_m1,region_.theta_second_tile_in_raw,region_.theta_second_tile_out_raw,region_.theta_second_tile_m0_raw,region_.theta_second_tile_m1_raw,region_.theta_q_base_cache_raw,region_.cnk_alpha,region_.cnk_range,region_.cnk_alpha_alpha,region_.cnk_alpha_range,region_.cnk_range_range,region_.k_alpha,region_.k_range,region_.k_alpha_alpha,region_.k_alpha_range,region_.k_range_range,region_.b_alpha,region_.b_range,region_.b_alpha_alpha,region_.b_alpha_range,region_.b_range_alpha,region_.b_range_range,region_.b_scratch0,region_.b_scratch1,region_.a_value,region_.a_alpha,region_.a_range,region_.a_alpha_alpha,region_.a_alpha_range,region_.a_range_alpha,region_.a_range_range,region_.a_scratch,region_.m_delta,region_.m_alpha,region_.m_range,region_.m_dd,region_.m_da,region_.m_dr,region_.m_aa,region_.m_ar,region_.m_ra,region_.m_rr,region_.potrf_work}) if(p) cudaFree(p); if(region_.potrf_info) cudaFree(region_.potrf_info); if(right_solve_factor_) cudaFree(right_solve_factor_); if(right_solve_rhs_) cudaFree(right_solve_rhs_); if(cublas_) cublasDestroy(cublas_); if(cusolver_) cusolverDnDestroy(cusolver_); if(stream_) cudaStreamDestroy(stream_); }
int GpuContext::device_id() const { return device_id_; } cudaStream_t GpuContext::stream() const { return stream_; } cublasHandle_t GpuContext::cublas() const { return cublas_; } cusolverDnHandle_t GpuContext::cusolver() const { return cusolver_; }
PProfile& GpuContext::profile() { return profile_; }
const RegionGpuState& GpuContext::region() const { return region_; }
long long GpuContext::raw_h2d_bytes() const { return region_.raw_h2d_bytes; }
long long GpuContext::parameter_h2d_bytes() const { return region_.parameter_h2d_bytes; }
long long GpuContext::host_compute_d2h_bytes() const { return region_.host_compute_d2h_bytes; }
long long GpuContext::result_d2h_bytes() const { return region_.result_d2h_bytes; }
long long GpuContext::cjj_second_allocation_count() const { return region_.cjj_second_allocation_count; }
long long GpuContext::b_derivative_allocation_count() const { return region_.b_derivative_allocation_count; }
long long GpuContext::b_first_derivative_allocation_count() const { return region_.b_first_derivative_allocation_count; }
long long GpuContext::b_pair_derivative_allocation_count() const { return region_.b_pair_derivative_allocation_count; }
long long GpuContext::a_derivative_allocation_count() const { return region_.a_derivative_allocation_count; }
long long GpuContext::m_derivative_allocation_count() const { return region_.m_derivative_allocation_count; }
long long GpuContext::mu_sigma_allocation_count() const { return region_.mu_sigma_allocation_count; }
long long GpuContext::delta_theta_workspace_allocation_count() const { return region_.delta_theta_workspace_allocation_count; }
long long GpuContext::delta_theta_derivative_workspace_allocation_count() const { return region_.delta_theta_derivative_workspace_allocation_count; }
long long GpuContext::delta_theta_reduction_allocation_count() const { return region_.delta_theta_reduction_allocation_count; }
long long GpuContext::global_k_correction_allocation_count() const { return region_.global_k_correction_allocation_count; }
void GpuContext::record_model_d2h_bytes(size_t bytes) { region_.result_d2h_bytes+=static_cast<long long>(bytes); }
void GpuContext::bind_region(const WorkerInput& worker, const Matrix& knots) {
  const int n=worker.sites.rows,m=knots.rows,q=worker.x.cols;
  if (worker.sites.cols!=2 || worker.z.rows!=n || worker.z.cols!=1 || worker.x.rows!=n || knots.cols!=2)
    throw std::runtime_error("invalid persistent region shape");
  const auto sites_hash=fingerprint(worker.sites.a), z_hash=fingerprint(worker.z.a), x_hash=fingerprint(worker.x.a), knots_hash=fingerprint(knots.a);
  const bool same=region_.initialized && region_.n==n && region_.m==m && region_.q==q &&
      region_.sites_fingerprint==sites_hash && region_.z_fingerprint==z_hash && region_.x_fingerprint==x_hash && region_.knots_fingerprint==knots_hash;
  if (same) return;
  check(cudaSetDevice(device_id_));
  const long long prior_raw_bytes=region_.raw_h2d_bytes;
  const long long prior_parameter_bytes=region_.parameter_h2d_bytes;
  const long long prior_host_compute_bytes=region_.host_compute_d2h_bytes;
  const long long prior_result_bytes=region_.result_d2h_bytes;
  if (region_.b_first) check(cudaFree(region_.b_first));
  if (region_.k_first) check(cudaFree(region_.k_first));
  if (region_.b_pair_first) check(cudaFree(region_.b_pair_first));
  if (region_.b_pair_second) check(cudaFree(region_.b_pair_second));
  if (region_.k_pair) check(cudaFree(region_.k_pair));
  if (region_.k_second) check(cudaFree(region_.k_second));
  for(double* p:{region_.sites,region_.z,region_.x_col_major,region_.knots,region_.cnk,region_.k,region_.chol_k,region_.b,region_.cjj,region_.cjj_alpha_alpha,region_.cjj_alpha_range,region_.cjj_range_range,region_.m_matrix,region_.chol_m,region_.bt_rinv_b,region_.producer_bt_minv_b,region_.cnk_alpha,region_.cnk_range,region_.cnk_alpha_alpha,region_.cnk_alpha_range,region_.cnk_range_range,region_.k_alpha,region_.k_range,region_.k_alpha_alpha,region_.k_alpha_range,region_.k_range_range,region_.b_alpha,region_.b_range,region_.b_alpha_alpha,region_.b_alpha_range,region_.b_range_alpha,region_.b_range_range,region_.b_scratch0,region_.b_scratch1,region_.a_value,region_.a_alpha,region_.a_range,region_.a_alpha_alpha,region_.a_alpha_range,region_.a_range_alpha,region_.a_range_range,region_.a_scratch,region_.m_delta,region_.m_alpha,region_.m_range,region_.m_dd,region_.m_da,region_.m_dr,region_.m_aa,region_.m_ar,region_.m_ra,region_.m_rr,region_.potrf_work}) if(p) check(cudaFree(p)); if(region_.potrf_info) check(cudaFree(region_.potrf_info));
  for (double* p : {region_.bk,region_.residual_e,region_.minv_b,region_.minv_e,region_.bt_minv_e,region_.rhs,region_.chol_diag,region_.scalar_checkpoint,region_.parameter_gamma,region_.parameter_mu,region_.parameter_sigma,region_.parameter_delta,region_.parameter_theta,region_.gamma_x_solved,region_.gamma_local,region_.gamma_xx,region_.delta_theta_r,region_.delta_theta_u,region_.delta_theta_w,region_.delta_theta_q,region_.delta_theta_q_intermediate,region_.delta_theta_b_zero,region_.delta_theta_r_first,region_.delta_theta_r_second,region_.delta_theta_q_first,region_.delta_theta_q_second,region_.delta_theta_q_scratch0,region_.delta_theta_q_scratch1,region_.delta_theta_reduction_scratch,region_.delta_theta_summary,region_.delta_theta_stream_partials,region_.delta_theta_theta_first_mm}) if (p) check(cudaFree(p));
  const long long prior_cjj_allocations=region_.cjj_second_allocation_count;
  const long long prior_b_allocations=region_.b_derivative_allocation_count;
  const long long prior_b_first_allocations=region_.b_first_derivative_allocation_count;
  const long long prior_b_pair_allocations=region_.b_pair_derivative_allocation_count;
  const long long prior_a_allocations=region_.a_derivative_allocation_count;
  const long long prior_m_allocations=region_.m_derivative_allocation_count;
  const long long prior_delta_theta_allocations=region_.delta_theta_workspace_allocation_count;
  const long long prior_delta_theta_derivative_allocations=region_.delta_theta_derivative_workspace_allocation_count;
  const long long prior_delta_theta_reduction_allocations=region_.delta_theta_reduction_allocation_count;
  region_={}; region_.raw_h2d_bytes=prior_raw_bytes; region_.parameter_h2d_bytes=prior_parameter_bytes; region_.host_compute_d2h_bytes=prior_host_compute_bytes; region_.result_d2h_bytes=prior_result_bytes; region_.cjj_second_allocation_count=prior_cjj_allocations; region_.b_derivative_allocation_count=prior_b_allocations; region_.b_first_derivative_allocation_count=prior_b_first_allocations; region_.b_pair_derivative_allocation_count=prior_b_pair_allocations; region_.a_derivative_allocation_count=prior_a_allocations; region_.m_derivative_allocation_count=prior_m_allocations; region_.delta_theta_workspace_allocation_count=prior_delta_theta_allocations; region_.delta_theta_derivative_workspace_allocation_count=prior_delta_theta_derivative_allocations; region_.delta_theta_reduction_allocation_count=prior_delta_theta_reduction_allocations; region_.n=n; region_.m=m; region_.q=q;
  check(cudaMalloc(&region_.sites,size_t(n)*2*sizeof(double))); check(cudaMalloc(&region_.z,size_t(n)*sizeof(double)));
  check(cudaMalloc(&region_.x_col_major,size_t(n)*q*sizeof(double))); check(cudaMalloc(&region_.knots,size_t(m)*2*sizeof(double)));
  check(cudaMalloc(&region_.cnk,size_t(n)*m*sizeof(double))); check(cudaMalloc(&region_.b,size_t(n)*m*sizeof(double)));
  check(cudaMalloc(&region_.k,size_t(m)*m*sizeof(double))); check(cudaMalloc(&region_.chol_k,size_t(m)*m*sizeof(double))); check(cudaMalloc(&region_.potrf_info,sizeof(int)));
  // The active device path factors M destructively in chol_m.  m_matrix is
  // only the short-lived B K B' assembly buffer and cjj is no longer an
  // allocation: chol_m first holds Cjj and is overwritten by M then POTRF.
  // Keeping all three was the dominant (and real) 15k lifetime violation.
  check(cudaMalloc(&region_.chol_m,size_t(n)*n*sizeof(double))); check(cudaMalloc(&region_.bt_rinv_b,size_t(m)*m*sizeof(double)));
  check(cudaMalloc(&region_.bk,size_t(n)*m*sizeof(double))); check(cudaMalloc(&region_.residual_e,size_t(n)*sizeof(double))); check(cudaMalloc(&region_.minv_b,size_t(n)*m*sizeof(double))); check(cudaMalloc(&region_.minv_e,size_t(n)*sizeof(double))); check(cudaMalloc(&region_.bt_minv_e,size_t(m)*sizeof(double))); check(cudaMalloc(&region_.rhs,size_t(n)*sizeof(double))); check(cudaMalloc(&region_.chol_diag,size_t(n)*sizeof(double))); check(cudaMalloc(&region_.scalar_checkpoint,4*sizeof(double))); check(cudaMalloc(&region_.parameter_gamma,size_t(q)*sizeof(double))); check(cudaMalloc(&region_.parameter_mu,size_t(m)*sizeof(double))); check(cudaMalloc(&region_.parameter_sigma,size_t(m)*m*sizeof(double))); check(cudaMalloc(&region_.parameter_delta,sizeof(double))); check(cudaMalloc(&region_.parameter_theta,3*sizeof(double))); check(cudaMalloc(&region_.gamma_x_solved,size_t(n)*q*sizeof(double))); check(cudaMalloc(&region_.gamma_local,size_t(q)*sizeof(double))); check(cudaMalloc(&region_.gamma_xx,size_t(q)*q*sizeof(double))); ++region_.mu_sigma_allocation_count;
  std::vector<double> xcol(size_t(n)*q); for(int i=0;i<n;++i) for(int j=0;j<q;++j) xcol[i+n*j]=worker.x(i,j);
  check(cudaMemcpyAsync(region_.sites,worker.sites.a.data(),size_t(n)*2*sizeof(double),cudaMemcpyHostToDevice,stream_));
  check(cudaMemcpyAsync(region_.z,worker.z.a.data(),size_t(n)*sizeof(double),cudaMemcpyHostToDevice,stream_));
  check(cudaMemcpyAsync(region_.x_col_major,xcol.data(),size_t(n)*q*sizeof(double),cudaMemcpyHostToDevice,stream_));
  check(cudaMemcpyAsync(region_.knots,knots.a.data(),size_t(m)*2*sizeof(double),cudaMemcpyHostToDevice,stream_)); check(cudaStreamSynchronize(stream_));
  region_.host_sites=worker.sites.a.data(); region_.host_z=worker.z.a.data(); region_.host_x=worker.x.a.data(); region_.host_knots=knots.a.data(); region_.sites_fingerprint=sites_hash; region_.z_fingerprint=z_hash; region_.x_fingerprint=x_hash; region_.knots_fingerprint=knots_hash; region_.raw_h2d_bytes+=size_t(n)*(3+q)*sizeof(double)+size_t(m)*2*sizeof(double); region_.initialized=true;
}
long long GpuContext::right_solve_allocation_count() const { return right_solve_allocation_count_; }
long long GpuContext::right_solve_factor_upload_count() const { return right_solve_factor_upload_count_; }
Matrix gpu_right_spd_solve(GpuContext&c,const Matrix&x,const Matrix&l){
  if(l.rows!=l.cols||x.cols!=l.rows)throw std::runtime_error("gpu right solve shape");
  check(cudaSetDevice(c.device_id())); const int n=l.rows,r=x.rows; const size_t factor_count=size_t(n)*n,rhs_count=size_t(n)*r;
  if(c.right_solve_factor_capacity_<factor_count){if(c.right_solve_factor_)check(cudaFree(c.right_solve_factor_));check(cudaMalloc(&c.right_solve_factor_,factor_count*sizeof(double)));c.right_solve_factor_capacity_=factor_count;++c.right_solve_allocation_count_;}
  if(c.right_solve_rhs_capacity_<rhs_count){if(c.right_solve_rhs_)check(cudaFree(c.right_solve_rhs_));check(cudaMalloc(&c.right_solve_rhs_,rhs_count*sizeof(double)));c.right_solve_rhs_capacity_=rhs_count;++c.right_solve_allocation_count_;}
  if(c.right_solve_factor_host_!=l.a){c.right_solve_factor_host_=l.a;std::vector<double> column_major(factor_count);for(int i=0;i<n;i++)for(int j=0;j<n;j++)column_major[i+n*j]=l(i,j);check(cudaMemcpyAsync(c.right_solve_factor_,column_major.data(),factor_count*sizeof(double),cudaMemcpyHostToDevice,c.stream()));++c.right_solve_factor_upload_count_;}
  c.right_solve_rhs_host_.resize(rhs_count);for(int i=0;i<n;i++)for(int j=0;j<r;j++)c.right_solve_rhs_host_[i+n*j]=x(j,i);
  check(cudaMemcpyAsync(c.right_solve_rhs_,c.right_solve_rhs_host_.data(),rhs_count*sizeof(double),cudaMemcpyHostToDevice,c.stream()));double one=1.;check(cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,r,&one,c.right_solve_factor_,n,c.right_solve_rhs_,n));check(cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,r,&one,c.right_solve_factor_,n,c.right_solve_rhs_,n));check(cudaMemcpyAsync(c.right_solve_rhs_host_.data(),c.right_solve_rhs_,rhs_count*sizeof(double),cudaMemcpyDeviceToHost,c.stream()));check(cudaStreamSynchronize(c.stream()));Matrix out(r,n);for(int i=0;i<n;i++)for(int j=0;j<r;j++)out(j,i)=c.right_solve_rhs_host_[i+n*j];return out;
}

LocalComputationResult compute_mu_sigma_local_legacy_oracle(GpuContext& context, const WorkerInput& worker, const Matrix& knots, const GlobalParameters& p) {
  PProfileScope profile_scope(context.profile(), "compute_mu_sigma_local");
  if (cached_local_matches(context, worker, knots, p)) return local_result_cache.result;
  context.bind_region(worker, knots); check(cudaSetDevice(context.device_id())); const int n=worker.sites.rows,m=knots.rows,q=worker.x.cols;
  double *ds{},*dk{},*dC{},*dK{},*dLK{},*dCjj{},*dBK{},*dM{},*dLM{},*dz{},*dx{},*dg{},*de{},*dY{},*dv{},*dA{},*dgv{},*dmu{},*dsigma{},*dr{},*du{},*dd{},*work{}; int*info{};
  // dC/dK/dLK are RegionGpuState-owned and must survive the local call.
  // All remaining allocations retain the old ownership/lifetime here.
  auto free_all=[&]{for(double*q:{dCjj,dM,dBK,dg,de,dY,dv,dgv,dmu,dsigma,dr,du,dd,work})if(q)cudaFree(q);if(info)cudaFree(info);};
  try { auto alloc=[&](double**q,size_t count){check(cudaMalloc(q,count*sizeof(double)));};
    // This is the active B path: Cnk, K, chol(K), and B are constructed in
    // the persistent region workspace, not reconstructed in this routine.
    const auto b_slice=compute_device_b_slice(context,p.nu2,p.alpha,p.range);
    const auto& region=context.region(); ds=region.sites;dk=region.knots;dz=region.z;dx=region.x_col_major;
    dC=const_cast<double*>(b_slice.b); dK=const_cast<double*>(b_slice.k); dLK=const_cast<double*>(b_slice.chol_k);
    dLM=region.chol_m; dA=region.bt_rinv_b;
    // Oracle-only compatibility storage is deliberately not part of the
    // resident region lifetime graph.
    alloc(&dCjj,size_t(n)*n);alloc(&dM,size_t(n)*n);alloc(&dBK,m*n);alloc(&dg,q);alloc(&de,n);alloc(&dY,n*m);alloc(&dv,n);alloc(&dgv,m);alloc(&dmu,m);alloc(&dsigma,m*m);alloc(&dr,n);alloc(&du,n);alloc(&dd,n);check(cudaMalloc(&info,sizeof(int)));
    check(cudaMemcpyAsync(dg,p.gamma.a.data(),q*sizeof(double),cudaMemcpyHostToDevice,context.stream()));check(cudaMemcpyAsync(dmu,p.mu.a.data(),m*sizeof(double),cudaMemcpyHostToDevice,context.stream()));check(cudaMemcpyAsync(dsigma,p.sigma.a.data(),m*m*sizeof(double),cudaMemcpyHostToDevice,context.stream()));
    matern<<<(n*n+63)/64,64,0,context.stream()>>>(ds,ds,dCjj,n,n,p.nu2,p.alpha,p.range);check(cudaGetLastError());
    int wm=0;check(cusolverDnDpotrf_bufferSize(context.cusolver(),CUBLAS_FILL_MODE_LOWER,n,dLM,n,&wm));alloc(&work,wm);int hinfo=-1;
    const double one=1.,zero=0.;check(cublasDgemm(context.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,m,&one,dK,m,dC,m,&zero,dBK,m));check(cublasDgemm(context.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,n,n,m,&one,dBK,m,dC,m,&zero,dM,n));make_m<<<(n*n+63)/64,64,0,context.stream()>>>(dCjj,dM,dM,n,p.delta);check(cudaGetLastError());check(cudaMemcpyAsync(dLM,dM,n*n*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));check(cusolverDnDpotrf(context.cusolver(),CUBLAS_FILL_MODE_LOWER,n,dLM,n,work,wm,info));check(cudaMemcpyAsync(&hinfo,info,sizeof(int),cudaMemcpyDeviceToHost,context.stream()));check(cudaStreamSynchronize(context.stream()));if(hinfo)throw std::runtime_error("M POTRF failed");
    residual<<<(n+63)/64,64,0,context.stream()>>>(dz,dx,dg,de,n,q);check(cudaGetLastError());check(cudaMemcpyAsync(dY,dC,n*m*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));check(cudaMemcpyAsync(dv,de,n*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));check(cublasDgeam(context.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,n,m,&one,dC,m,&zero,dC,n,dY,n));check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,m,&one,dLM,n,dY,n));check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,m,&one,dLM,n,dY,n));check(cublasDscal(context.cublas(),n*m,&p.delta,dY,1));check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,1,&one,dLM,n,dv,n));check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,1,&one,dLM,n,dv,n));check(cublasDscal(context.cublas(),n,&p.delta,dv,1));
    check(cublasDgemm(context.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,m,m,n,&one,dC,m,dY,n,&zero,dA,m));check(cublasDgemv(context.cublas(),CUBLAS_OP_N,m,n,&one,dC,m,dv,1,&zero,dgv,1));double q=0;check(cublasDdot(context.cublas(),n,de,1,dv,1,&q));diag<<<(n+63)/64,64,0,context.stream()>>>(dLM,dd,n);check(cudaGetLastError());
    Matrix cstar(n,m);check(cudaMemcpyAsync(cstar.a.data(),dC,n*m*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaStreamSynchronize(context.stream()));check(cublasDgemv(context.cublas(),CUBLAS_OP_T,m,n,&one,dC,m,dmu,1,&zero,dr,1));subtract_vec<<<(n+63)/64,64,0,context.stream()>>>(dr,de,n);check(cudaGetLastError());check(cudaMemcpyAsync(du,dr,n*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,1,&one,dLM,n,du,n));check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,1,&one,dLM,n,du,n));check(cublasDscal(context.cublas(),n,&p.delta,du,1));double quad=0,trace=0;check(cublasDdot(context.cublas(),n,dr,1,du,1,&quad));check(cublasDdot(context.cublas(),m*m,dA,1,dsigma,1,&trace));LocalComputationResult out;out.cstar=cstar;out.B=Matrix(n,m);out.M=Matrix(n,n);out.chol_M=Matrix(n,n);out.e=Matrix(n,1);out.rinv_B=Matrix(n,m);out.bt_rinv_B=Matrix(m,m);out.rinv_e=Matrix(n,1);out.bt_rinv_e=Matrix(m,1);std::vector<double>hd(n);check(cudaMemcpyAsync(out.B.a.data(),dC,n*m*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(out.M.a.data(),dM,n*n*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(out.chol_M.a.data(),dLM,n*n*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(out.e.a.data(),de,n*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(out.rinv_B.a.data(),dY,n*m*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(out.rinv_e.a.data(),dv,n*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(out.bt_rinv_B.a.data(),dA,m*m*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(out.bt_rinv_e.a.data(),dgv,m*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(hd.data(),dd,n*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaStreamSynchronize(context.stream()));context.record_model_d2h_bytes(size_t(n)*m*sizeof(double)*3+size_t(n)*n*sizeof(double)*2+size_t(m)*m*sizeof(double)+size_t(n)*sizeof(double)*3+size_t(m)*sizeof(double));out.logdet_R=0;for(double v:hd)out.logdet_R+=2*std::log(v);out.logdet_R-=n*std::log(p.delta);out.et_rinv_e=q;out.rinv_B=transpose(out.rinv_B);Matrix lrow=transpose(out.chol_M);for(int i=0;i<n;i++)for(int j=i+1;j<n;j++)lrow(i,j)=0;out.chol_M=lrow;out.summary.mu=out.bt_rinv_e;for(double&v:out.summary.mu.a)v=-v/p.delta;out.summary.sigma=out.bt_rinv_B;for(double&v:out.summary.sigma.a)v/=p.delta;out.local_objective=out.logdet_R+quad+trace;free_all();cache_local_result(context, worker, knots, p, out);return out;
  } catch(...) {free_all();throw;}
}

DeviceMuSigmaLocalState build_device_mu_sigma_local(GpuContext& context, const WorkerInput& worker, const Matrix& knots, const GlobalParameters& p) {
  PProfileScope profile_scope(context.profile(), "build_device_mu_sigma_local");
  context.bind_region(worker, knots); check(cudaSetDevice(context.device_id())); auto& r=const_cast<RegionGpuState&>(context.region()); const int n=r.n,m=r.m,q=r.q;
  const auto gh=fingerprint(p.gamma.a), mh=fingerprint(p.mu.a), sh=fingerprint(p.sigma.a);
  const bool invalid=!r.parameters_valid, gamma_changed=r.parameter_gamma_fingerprint!=gh,
      mu_changed=r.parameter_mu_fingerprint!=mh, sigma_changed=r.parameter_sigma_fingerprint!=sh,
      delta_changed=r.parameter_delta_value!=p.delta, alpha_changed=r.parameter_alpha_value!=p.alpha,
      range_changed=r.parameter_range_value!=p.range, nu_changed=r.parameter_nu2!=p.nu2;
  const bool parameters_changed=invalid || gamma_changed || mu_changed || sigma_changed ||
      delta_changed || alpha_changed || range_changed || nu_changed;
  if(parameters_changed){check(cudaMemcpyAsync(r.parameter_gamma,p.gamma.a.data(),size_t(q)*sizeof(double),cudaMemcpyHostToDevice,context.stream())); check(cudaMemcpyAsync(r.parameter_mu,p.mu.a.data(),size_t(m)*sizeof(double),cudaMemcpyHostToDevice,context.stream())); check(cudaMemcpyAsync(r.parameter_sigma,p.sigma.a.data(),size_t(m)*m*sizeof(double),cudaMemcpyHostToDevice,context.stream())); check(cudaMemcpyAsync(r.parameter_delta,&p.delta,sizeof(double),cudaMemcpyHostToDevice,context.stream()));r.parameter_h2d_bytes+=static_cast<long long>((size_t(q)+size_t(m)+size_t(m)*m+1)*sizeof(double));r.parameter_gamma_fingerprint=gh;r.parameter_mu_fingerprint=mh;r.parameter_sigma_fingerprint=sh;r.parameter_delta_value=p.delta;r.parameter_alpha_value=p.alpha;r.parameter_range_value=p.range;r.parameter_nu2=p.nu2;r.parameters_valid=true;}
  if (r.producer_state_cache_active && r.producer_state_cache_valid && !parameters_changed) {
    ++r.producer_state_cache_hit_count;
    return {n,m,q,r.b,nullptr,r.chol_m,r.minv_b,r.residual_e,r.minv_e,
            r.bt_minv_e,r.producer_bt_minv_b,r.chol_diag,r.parameter_gamma,
            r.parameter_mu,r.parameter_sigma,r.parameter_delta,p.delta,r.scalar_checkpoint};
  }
  ++r.producer_state_build_count;
  // chol_m is overwritten below.  Clear the separately-owned inverse cache
  // before that happens, even though its allocation stays reusable.
  r.local_dense_w_valid=false;
  const auto bs=compute_device_b_slice(context,p.nu2,p.alpha,p.range); const double one=1.,zero=0.;
  // Phase 1: Cjj lives in the eventual factor allocation.  Phase 2 creates
  // BKB' in the sole transient n*n buffer and overwrites chol_m with M.
  // The transient is released before the POTRF workspace becomes live.
  matern<<<(n*n+63)/64,64,0,context.stream()>>>(r.sites,r.sites,r.chol_m,n,n,p.nu2,p.alpha,p.range); check(cudaGetLastError());
  if (!r.m_matrix) check(cudaMalloc(&r.m_matrix,size_t(n)*n*sizeof(double)));
  const bool reduced_product=
      context.precision_policy().mode==ArithmeticPrecision::fp32 &&
      reduced_bkb(context,r,bs.k,bs.b,n,m);
  if(!reduced_product){check(cublasDgemm(context.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,m,&one,bs.k,m,bs.b,m,&zero,r.bk,m));check(cublasDgemm(context.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,n,n,m,&one,r.bk,m,bs.b,m,&zero,r.m_matrix,n));}
  make_m<<<(n*n+63)/64,64,0,context.stream()>>>(r.chol_m,r.m_matrix,r.chol_m,n,p.delta); check(cudaGetLastError());
  check(cudaStreamSynchronize(context.stream()));
  int work_size=0; check(cusolverDnDpotrf_bufferSize(context.cusolver(),CUBLAS_FILL_MODE_LOWER,n,r.chol_m,n,&work_size)); if(r.potrf_work_capacity<size_t(work_size)){if(r.potrf_work)check(cudaFree(r.potrf_work));check(cudaMalloc(&r.potrf_work,size_t(work_size)*sizeof(double)));r.potrf_work_capacity=work_size;++r.mu_sigma_allocation_count;} check(cusolverDnDpotrf(context.cusolver(),CUBLAS_FILL_MODE_LOWER,n,r.chol_m,n,r.potrf_work,work_size,r.potrf_info)); int info=-1;check(cudaMemcpyAsync(&info,r.potrf_info,sizeof(int),cudaMemcpyDeviceToHost,context.stream()));check(cudaStreamSynchronize(context.stream()));if(info&&reduced_product&&context.precision_policy().promote_on_failure){check(cublasDgemm(context.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,m,&one,bs.k,m,bs.b,m,&zero,r.bk,m));check(cublasDgemm(context.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,n,n,m,&one,r.bk,m,bs.b,m,&zero,r.m_matrix,n));make_m<<<(n*n+63)/64,64,0,context.stream()>>>(r.chol_m,r.m_matrix,r.chol_m,n,p.delta);check(cudaGetLastError());check(cusolverDnDpotrf(context.cusolver(),CUBLAS_FILL_MODE_LOWER,n,r.chol_m,n,r.potrf_work,work_size,r.potrf_info));check(cudaMemcpyAsync(&info,r.potrf_info,sizeof(int),cudaMemcpyDeviceToHost,context.stream()));check(cudaStreamSynchronize(context.stream()));++r.precision_promotions;}check(cudaFree(r.m_matrix));r.m_matrix=nullptr;if(info)throw std::runtime_error("M POTRF failed");
  residual<<<(n+63)/64,64,0,context.stream()>>>(r.z,r.x_col_major,r.parameter_gamma,r.residual_e,n,q);check(cudaGetLastError()); check(cublasDgeam(context.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,n,m,&one,bs.b,m,&zero,bs.b,n,r.minv_b,n)); check(cudaMemcpyAsync(r.minv_e,r.residual_e,size_t(n)*sizeof(double),cudaMemcpyDeviceToDevice,context.stream())); check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,m,&one,r.chol_m,n,r.minv_b,n));check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,m,&one,r.chol_m,n,r.minv_b,n));check(cublasDscal(context.cublas(),n*m,&p.delta,r.minv_b,1));check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,1,&one,r.chol_m,n,r.minv_e,n));check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,1,&one,r.chol_m,n,r.minv_e,n));check(cublasDscal(context.cublas(),n,&p.delta,r.minv_e,1));
  check(cublasDgemm(context.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,m,m,n,&one,bs.b,m,r.minv_b,n,&zero,r.bt_rinv_b,m));check(cublasDgemv(context.cublas(),CUBLAS_OP_N,m,n,&one,bs.b,m,r.minv_e,1,&zero,r.bt_minv_e,1));check(cublasSetPointerMode(context.cublas(),CUBLAS_POINTER_MODE_DEVICE));check(cublasDdot(context.cublas(),n,r.residual_e,1,r.minv_e,1,r.scalar_checkpoint));check(cublasSetPointerMode(context.cublas(),CUBLAS_POINTER_MODE_HOST));local_logdet<<<1,1,0,context.stream()>>>(r.chol_m,n,p.delta,r.scalar_checkpoint);check(cudaGetLastError());
  check(cublasDgemv(context.cublas(),CUBLAS_OP_T,m,n,&one,bs.b,m,r.parameter_mu,1,&zero,r.rhs,1));subtract_vec<<<(n+63)/64,64,0,context.stream()>>>(r.rhs,r.residual_e,n);check(cudaGetLastError());check(cudaMemcpyAsync(r.minv_e,r.rhs,size_t(n)*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,1,&one,r.chol_m,n,r.minv_e,n));check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,1,&one,r.chol_m,n,r.minv_e,n));check(cublasDscal(context.cublas(),n,&p.delta,r.minv_e,1));check(cublasSetPointerMode(context.cublas(),CUBLAS_POINTER_MODE_DEVICE));check(cublasDdot(context.cublas(),n,r.rhs,1,r.minv_e,1,r.scalar_checkpoint+2));check(cublasDdot(context.cublas(),m*m,r.bt_rinv_b,1,r.parameter_sigma,1,r.scalar_checkpoint+3));check(cublasSetPointerMode(context.cublas(),CUBLAS_POINTER_MODE_HOST));finish_local_objective<<<1,1,0,context.stream()>>>(r.scalar_checkpoint);check(cudaGetLastError());
  // Restore M^-1 e after objective scratch reuse.
  check(cudaMemcpyAsync(r.minv_e,r.residual_e,size_t(n)*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,1,&one,r.chol_m,n,r.minv_e,n));check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,1,&one,r.chol_m,n,r.minv_e,n));check(cublasDscal(context.cublas(),n,&p.delta,r.minv_e,1));
  if (r.producer_state_cache_active) {
    if (!r.producer_bt_minv_b)
      check(cudaMalloc(&r.producer_bt_minv_b,size_t(m)*m*sizeof(double)));
    check(cudaMemcpyAsync(r.producer_bt_minv_b,r.bt_rinv_b,size_t(m)*m*sizeof(double),
                          cudaMemcpyDeviceToDevice,context.stream()));
    r.producer_state_cache_valid=true;
  }
  return {n,m,q,bs.b,nullptr,r.chol_m,r.minv_b,r.residual_e,r.minv_e,r.bt_minv_e,
          r.producer_state_cache_active ? r.producer_bt_minv_b : r.bt_rinv_b,
          r.chol_diag,r.parameter_gamma,r.parameter_mu,r.parameter_sigma,r.parameter_delta,p.delta,r.scalar_checkpoint};
}

DeviceDeltaThetaWorkspace build_device_delta_theta_workspace(
    GpuContext& context, const DeviceMuSigmaLocalState& state) {
  check(cudaSetDevice(context.device_id()));
  auto& region=const_cast<RegionGpuState&>(context.region());
  if (!region.initialized || state.n!=region.n || state.m!=region.m ||
      state.B!=region.b ||
      state.chol_M!=region.chol_m || state.e!=region.residual_e ||
      state.mu!=region.parameter_mu || state.sigma!=region.parameter_sigma)
    throw std::runtime_error("delta/theta workspace state is not owned by this region");
  const int n=state.n,m=state.m;
  if (!region.delta_theta_r) {
    check(cudaMalloc(&region.delta_theta_r,size_t(n)*sizeof(double)));
    check(cudaMalloc(&region.delta_theta_u,size_t(n)*sizeof(double)));
    check(cudaMalloc(&region.delta_theta_w,size_t(n)*n*sizeof(double)));
    check(cudaMalloc(&region.delta_theta_q,size_t(n)*n*sizeof(double)));
    check(cudaMalloc(&region.delta_theta_q_intermediate,size_t(n)*m*sizeof(double)));
    ++region.delta_theta_workspace_allocation_count;
  }
  const double one=1.,zero=0.;
  // B is row-major, so cuBLAS sees B' as an m-by-n column-major matrix.
  // This is the legacy r=B mu-e operation, in that order.
  check(cublasDgemv(context.cublas(),CUBLAS_OP_T,m,n,&one,state.B,m,state.mu,1,
                    &zero,region.delta_theta_r,1));
  subtract_vec<<<(n+63)/64,64,0,context.stream()>>>(region.delta_theta_r,state.e,n);
  check(cudaGetLastError());
  // u=M^-1 r, followed by W=M^-1 using the same two triangular solves as
  // the legacy solve_spd definition.
  check(cudaMemcpyAsync(region.delta_theta_u,region.delta_theta_r,size_t(n)*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()));
  check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,1,&one,state.chol_M,n,region.delta_theta_u,n));
  check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,1,&one,state.chol_M,n,region.delta_theta_u,n));
  set_identity<<<(n*n+63)/64,64,0,context.stream()>>>(region.delta_theta_w,n);
  check(cudaGetLastError());
  check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,n,&one,state.chol_M,n,region.delta_theta_w,n));
  check(cublasDtrsm(context.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,n,&one,state.chol_M,n,region.delta_theta_w,n));
  // Q=B Sigma B'.  The intermediate is (B Sigma)' in cuBLAS layout.
  check(cublasDgemm(context.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,m,&one,
                    state.sigma,m,state.B,m,&zero,region.delta_theta_q_intermediate,m));
  check(cublasDgemm(context.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,n,n,m,&one,
                    region.delta_theta_q_intermediate,m,state.B,m,&zero,
                    region.delta_theta_q,n));
  return {n,m,region.delta_theta_r,region.delta_theta_u,region.delta_theta_w,
          region.delta_theta_q,region.delta_theta_q_intermediate};
}

DeltaThetaWorkspaceDownload download_device_delta_theta_workspace_for_test(
    GpuContext& context, const DeviceDeltaThetaWorkspace& workspace) {
  DeltaThetaWorkspaceDownload out{Matrix(workspace.n,1),Matrix(workspace.n,1),
                                  Matrix(workspace.n,workspace.n),Matrix(workspace.n,workspace.n)};
  check(cudaMemcpyAsync(out.r.a.data(),workspace.r,size_t(workspace.n)*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));
  check(cudaMemcpyAsync(out.u.a.data(),workspace.u,size_t(workspace.n)*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));
  check(cudaMemcpyAsync(out.W.a.data(),workspace.W,size_t(workspace.n)*workspace.n*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));
  check(cudaMemcpyAsync(out.Q.a.data(),workspace.Q,size_t(workspace.n)*workspace.n*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));
  check(cudaStreamSynchronize(context.stream()));
  context.record_model_d2h_bytes(size_t(2*workspace.n+2*workspace.n*workspace.n)*sizeof(double));
  return out;
}

namespace {
// out += left * sigma * right'.  Row-major inputs are consumed through the
// same transpose interpretation used by the base Q=B Sigma B' builder.
void q_product_add(GpuContext& context, const double* left, const double* sigma,
                   const double* right, double* out, double* intermediate,
                   double* term, int n, int m) {
  const double one=1.,zero=0.;
  check(cublasDgemm(context.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,m,&one,
                    sigma,m,left,m,&zero,intermediate,m));
  check(cublasDgemm(context.cublas(),CUBLAS_OP_T,CUBLAS_OP_N,n,n,m,&one,
                    intermediate,m,right,m,&zero,term,n));
  check(cublasDaxpy(context.cublas(),n*n,&one,term,1,out,1));
}
}

DeviceDeltaThetaDerivativeWorkspace build_device_delta_theta_derivative_workspace(
    GpuContext& context, const DeviceMuSigmaLocalState& state) {
  check(cudaSetDevice(context.device_id()));
  auto& region=const_cast<RegionGpuState&>(context.region());
  if (!region.initialized || state.n!=region.n || state.m!=region.m ||
      state.B!=region.b || state.mu!=region.parameter_mu ||
      state.sigma!=region.parameter_sigma)
    throw std::runtime_error("delta/theta derivative workspace state is not owned by this region");
  const int n=state.n,m=state.m; const size_t nm=size_t(n)*m,nn=size_t(n)*n;
  const double one=1.,zero=0.;
  // The B bundle is persistent and solve-based; no host reconstruction or
  // model-sized transfer occurs here.
  const auto b=compute_device_b_derivatives(context,region.parameter_nu2,
                                            region.parameter_alpha_value,
                                            region.parameter_range_value);
  if (!region.delta_theta_b_zero) {
    auto alloc=[&](double** p,size_t count){ check(cudaMalloc(p,count*sizeof(double))); };
    alloc(&region.delta_theta_b_zero,nm); alloc(&region.delta_theta_r_first,3*size_t(n));
    alloc(&region.delta_theta_r_second,9*size_t(n)); alloc(&region.delta_theta_q_first,3*nn);
    alloc(&region.delta_theta_q_second,9*nn); alloc(&region.delta_theta_q_scratch0,nm);
    alloc(&region.delta_theta_q_scratch1,nn); ++region.delta_theta_derivative_workspace_allocation_count;
  }
  set_zero<<<(nm+255)/256,256,0,context.stream()>>>(region.delta_theta_b_zero,nm);
  check(cudaGetLastError());
  DeviceDeltaThetaDerivativeWorkspace out; out.n=n; out.m=m;
  out.B_i={region.delta_theta_b_zero,b.alpha,b.range};
  out.B_ij={{{region.delta_theta_b_zero,region.delta_theta_b_zero,region.delta_theta_b_zero},
             {region.delta_theta_b_zero,b.alpha_alpha,b.alpha_range},
             {region.delta_theta_b_zero,b.range_alpha,b.range_range}}};
  for(int i=0;i<3;++i) {
    double* ri=region.delta_theta_r_first+size_t(i)*n;
    check(cublasDgemv(context.cublas(),CUBLAS_OP_T,m,n,&one,out.B_i[i],m,state.mu,1,&zero,ri,1));
    out.r_i[i]=ri;
    double* qi=region.delta_theta_q_first+size_t(i)*nn;
    set_zero<<<(nn+255)/256,256,0,context.stream()>>>(qi,nn); check(cudaGetLastError());
    // Exact legacy q_first order: Bi Sigma B' then B Sigma Bi'.
    q_product_add(context,out.B_i[i],state.sigma,state.B,qi,region.delta_theta_q_scratch0,region.delta_theta_q_scratch1,n,m);
    q_product_add(context,state.B,state.sigma,out.B_i[i],qi,region.delta_theta_q_scratch0,region.delta_theta_q_scratch1,n,m);
    out.Q_i[i]=qi;
    for(int j=0;j<3;++j) {
      double* rij=region.delta_theta_r_second+size_t(3*i+j)*n;
      check(cublasDgemv(context.cublas(),CUBLAS_OP_T,m,n,&one,out.B_ij[i][j],m,state.mu,1,&zero,rij,1));
      out.r_ij[i][j]=rij;
      double* qij=region.delta_theta_q_second+size_t(3*i+j)*nn;
      set_zero<<<(nn+255)/256,256,0,context.stream()>>>(qij,nn); check(cudaGetLastError());
      // Exact legacy q_second order: Bij Sigma B', B Sigma Bij',
      // Bi Sigma Bj', Bj Sigma Bi'.
      q_product_add(context,out.B_ij[i][j],state.sigma,state.B,qij,region.delta_theta_q_scratch0,region.delta_theta_q_scratch1,n,m);
      q_product_add(context,state.B,state.sigma,out.B_ij[i][j],qij,region.delta_theta_q_scratch0,region.delta_theta_q_scratch1,n,m);
      q_product_add(context,out.B_i[i],state.sigma,out.B_i[j],qij,region.delta_theta_q_scratch0,region.delta_theta_q_scratch1,n,m);
      q_product_add(context,out.B_i[j],state.sigma,out.B_i[i],qij,region.delta_theta_q_scratch0,region.delta_theta_q_scratch1,n,m);
      out.Q_ij[i][j]=qij;
    }
  }
  return out;
}

DeltaThetaDerivativeWorkspaceDownload
download_device_delta_theta_derivative_workspace_for_test(
    GpuContext& context, const DeviceDeltaThetaDerivativeWorkspace& workspace) {
  if (workspace.n<=0 || workspace.m<=0 || !workspace.B_i[1])
    throw std::runtime_error("invalid delta/theta derivative workspace download");
  DeltaThetaDerivativeWorkspaceDownload out;
  const size_t nm=size_t(workspace.n)*workspace.m, n=size_t(workspace.n), nn=n*n;
  auto load=[&](Matrix& host,const double* device,int rows,int cols,size_t count) {
    host=Matrix(rows,cols); check(cudaMemcpyAsync(host.a.data(),device,count*sizeof(double),cudaMemcpyDeviceToHost,context.stream())); };
  for(int i=0;i<3;++i) { load(out.B_i[i],workspace.B_i[i],workspace.n,workspace.m,nm); load(out.r_i[i],workspace.r_i[i],workspace.n,1,n); load(out.Q_i[i],workspace.Q_i[i],workspace.n,workspace.n,nn); }
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) { load(out.B_ij[i][j],workspace.B_ij[i][j],workspace.n,workspace.m,nm); load(out.r_ij[i][j],workspace.r_ij[i][j],workspace.n,1,n); load(out.Q_ij[i][j],workspace.Q_ij[i][j],workspace.n,workspace.n,nn); }
  check(cudaStreamSynchronize(context.stream()));
  context.record_model_d2h_bytes((3*nm+3*n+3*nn+9*nm+9*n+9*nn)*sizeof(double));
  return out;
}

LocalComputationResult download_device_mu_sigma_local(GpuContext& context, const DeviceMuSigmaLocalState& s) {
  LocalComputationResult out; out.cstar=Matrix(s.n,s.m);out.B=Matrix(s.n,s.m);out.M=Matrix(s.n,s.n);out.chol_M=Matrix(s.n,s.n);out.e=Matrix(s.n,1);out.rinv_B=Matrix(s.n,s.m);out.rinv_e=Matrix(s.n,1);out.bt_rinv_e=Matrix(s.m,1);out.bt_rinv_B=Matrix(s.m,s.m);double scalar[4];check(cudaMemcpyAsync(out.B.a.data(),s.B,size_t(s.n)*s.m*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(out.cstar.a.data(),s.B,size_t(s.n)*s.m*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(out.chol_M.a.data(),s.chol_M,size_t(s.n)*s.n*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(out.e.a.data(),s.e,size_t(s.n)*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(out.rinv_B.a.data(),s.minv_B,size_t(s.n)*s.m*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(out.rinv_e.a.data(),s.minv_e,size_t(s.n)*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(out.bt_rinv_e.a.data(),s.bt_minv_e,size_t(s.m)*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(out.bt_rinv_B.a.data(),s.bt_minv_B,size_t(s.m)*s.m*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaMemcpyAsync(scalar,s.scalar_checkpoint,4*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));check(cudaStreamSynchronize(context.stream()));context.record_model_d2h_bytes(size_t(s.n)*s.m*sizeof(double)*3+size_t(s.n)*s.n*sizeof(double)+size_t(s.m)*s.m*sizeof(double)+size_t(s.n)*sizeof(double)*2+size_t(s.m)*sizeof(double)+4*sizeof(double));out.rinv_B=transpose(out.rinv_B);Matrix lrow=transpose(out.chol_M);for(int i=0;i<s.n;i++)for(int j=i+1;j<s.n;j++)lrow(i,j)=0;out.chol_M=lrow;out.M=matmul(out.chol_M,transpose(out.chol_M));out.et_rinv_e=scalar[0];out.logdet_R=scalar[1];out.local_objective=scalar[3];out.summary.mu=out.bt_rinv_e;for(double&v:out.summary.mu.a)v=-v/s.delta;out.summary.sigma=out.bt_rinv_B;for(double&v:out.summary.sigma.a)v/=s.delta;return out;
}

LocalComputationResult compute_mu_sigma_local(GpuContext& context, const WorkerInput& worker, const Matrix& knots, const GlobalParameters& p) { if(cached_local_matches(context,worker,knots,p))return local_result_cache.result;auto out=download_device_mu_sigma_local(context,build_device_mu_sigma_local(context,worker,knots,p));cache_local_result(context,worker,knots,p,out);return out; }
}
