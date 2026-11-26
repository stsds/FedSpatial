#include "fedspatial/gpu_context.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace fedspatial;

namespace {
struct Error { double abs{}, rel{}; void add(double x, double y) { const double e=std::abs(x-y); abs=std::max(abs,e); rel=std::max(rel,e/std::max(1.,std::abs(y))); } };
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
Matrix knots() { Matrix k(2,2); k(0,0)=k(0,1)=0.; k(1,0)=k(1,1)=1.5; return k; }
WorkerInput worker() { Matrix s(3,2),z(3,1),x(3,1); const double a[]={0,0,1,.2,.3,1.1},b[]={1.2,-.4,.7}; std::copy(a,a+6,s.a.begin()); std::copy(b,b+3,z.a.begin()); for(double& v:x.a)v=1.; return {s,z,x}; }
GlobalParameters parameters() { Matrix g(1,1),m(2,1),s(2,2); g(0,0)=.15;m(0,0)=.2;m(1,0)=-.1;s(0,0)=.7;s(0,1)=s(1,0)=.05;s(1,1)=.4;return {g,m,s,1.7,1.2,.8,3}; }
double symmetry(const std::array<double,13>& x) { return std::max({std::abs(x[5]-x[7]),std::abs(x[6]-x[10]),std::abs(x[9]-x[11])}); }
double matrix_error(const Matrix& x, const Matrix& y) { double e=0.; for(size_t i=0;i<x.a.size();++i) e=std::max(e,std::abs(x.a[i]-y.a[i])); return e; }
}

int main() { try {
  GpuContext context(0); const auto k=knots(); const auto p=parameters();
  // Establish precisely the resident K/derivative/parameter state used by the
  // active evaluator.  Its compact output is discarded.
  (void)compute_device_delta_theta_summary(context, worker(), k, p);
  Matrix legacy_g,legacy_h; compute_legacy_global_k_correction_for_test(context,k,p,&legacy_g,&legacy_h);
  std::array<double,13> pre{{7., .125,-.25,.5, 1.,2.,3.,2.,5.,6.,3.,6.,9.}};
  std::array<double,13> expected=pre;
  for(int i=0;i<3;++i) { expected[1+i]+=legacy_g(i,0); for(int j=0;j<3;++j) expected[4+3*i+j]+=legacy_h(i,j); }
  double* device{}; check(cudaMalloc(&device,13*sizeof(double)));
  check(cudaMemcpyAsync(device,pre.data(),13*sizeof(double),cudaMemcpyHostToDevice,context.stream()));
  apply_device_global_k_correction(context,device);
  std::array<double,13> got{}; check(cudaMemcpyAsync(got.data(),device,13*sizeof(double),cudaMemcpyDeviceToHost,context.stream())); check(cudaStreamSynchronize(context.stream()));
  // The correction leaves its fixed-order RR intermediates in slots 0..9
  // and borrows the streamed inverse-column buffer for W mu.  Audit them
  // against the corresponding legacy
  // ordered matmul tree; this catches transpose/aliasing errors separately
  // from the final scalar contractions.
  const int n=k.rows; const size_t nn=size_t(n)*n;
  std::array<Matrix,10> scratch{}; for(Matrix& x:scratch) x=Matrix(n,n);
  Matrix device_u(n,1);
  const auto& rs=context.region();
  for(int q=0;q<10;++q) check(cudaMemcpyAsync(scratch[q].a.data(),rs.delta_theta_reduction_scratch+q*nn,nn*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));
  check(cudaMemcpyAsync(device_u.a.data(),rs.delta_theta_r,n*sizeof(double),cudaMemcpyDeviceToHost,context.stream()));
  check(cudaStreamSynchronize(context.stream()));
  const auto d=matern_with_derivatives(context,k,k,p.nu2,p.alpha,p.range);
  const Matrix w=solve_spd(d.value,identity(n)); const Matrix u=matmul(w,p.mu);
  const Matrix wk=matmul(w,d.range), wkw=matmul(wk,w), wkwk=matmul(wkw,d.range);
  const std::array<Matrix,10> legacy_scratch{{cholesky_lower(d.value),w,
    wkwk,matmul(matmul(w,d.rr),w),matmul(matmul(d.range,w),d.range),matmul(matmul(d.range,w),d.range),
    matmul(wkwk,w),matmul(wkwk,w),wkwk,wkw}};
  double scratch_max=matrix_error(device_u,u); for(int q=0;q<10;++q) scratch_max=std::max(scratch_max,matrix_error(scratch[q],legacy_scratch[q]));
  std::cout<<"GLOBAL_K_RR_INTERMEDIATE_MAX_ABS="<<scratch_max<<'\n';
  // A repeated application from the identical compact input must be bitwise
  // stable and must not depend on scratch contents left by the first call.
  std::array<double,13> repeat{}; check(cudaMemcpyAsync(device,pre.data(),13*sizeof(double),cudaMemcpyHostToDevice,context.stream())); apply_device_global_k_correction(context,device); check(cudaMemcpyAsync(repeat.data(),device,13*sizeof(double),cudaMemcpyDeviceToHost,context.stream())); check(cudaStreamSynchronize(context.stream())); check(cudaFree(device));
  Error correction,post; for(int i=0;i<3;++i) correction.add(got[1+i]-pre[1+i],legacy_g(i,0)); for(int i=0;i<9;++i) correction.add(got[4+i]-pre[4+i],legacy_h.a[i]); for(int i=0;i<13;++i) { post.add(got[i],expected[i]); std::cout<<std::setprecision(17)<<"ENTRY="<<i<<" PRE="<<pre[i]<<" CORRECTION="<<(got[i]-pre[i])<<" POST="<<got[i]<<" EXPECTED="<<expected[i]<<'\n'; }
  const bool repeat_ok=std::memcmp(got.data(),repeat.data(),sizeof(got))==0;
  const double sym=std::max(symmetry(got),symmetry(expected));
  // Scaling audit: local sums scale with J, while the global correction is
  // added once; order is fixed by deterministic_worker_order_mean.
  Matrix a(1,1),b(1,1),c(1,1); a(0,0)=1e16;b(0,0)=-1e16;c(0,0)=1.;
  const bool order=deterministic_worker_order_mean({a,b,c})(0,0)==1./3. && deterministic_worker_order_mean({a,c,b})(0,0)!=1./3.;
  std::cout<<"GLOBAL_K_CORRECTION_MAX_ABS="<<correction.abs<<" MAX_REL="<<correction.rel<<'\n'
           <<"GLOBAL_K_POST_MAX_ABS="<<post.abs<<" MAX_REL="<<post.rel<<'\n'
           <<"GLOBAL_K_SYMMETRY_MAX_ABS="<<sym<<'\n'
           <<"GLOBAL_K_REPEAT_BYTE_DETERMINISM="<<(repeat_ok?"PASS":"FAIL")<<'\n'
           <<"GLOBAL_K_WORKER_ORDER_AUDIT="<<(order?"PASS":"FAIL")<<'\n'
           <<"GLOBAL_K_CORRECTION_PARITY="<<(correction.abs<=1e-10&&post.abs<=1e-10&&sym<=1e-12&&repeat_ok&&order?"PASS":"FAIL")<<'\n';
  return correction.abs<=1e-10&&post.abs<=1e-10&&sym<=1e-12&&repeat_ok&&order?0:1;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; } }
