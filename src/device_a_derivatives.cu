#include "fedspatial/gpu_context.hpp"

#include <stdexcept>

namespace fedspatial {
namespace {
void check_a(cudaError_t value) { if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value)); }
void check_a(cublasStatus_t value) { if (value != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS A derivative failure"); }

// Initialize first derivatives from Cjj.  Second derivatives are supplied by
// the persistent Cjj derivative bundle, avoiding its former compatibility D2H.
__global__ void cjj_value_first_derivatives(const double* sites, int n, int nu2, double a,
                                            double r, double* value, double* da, double* dr) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n*n) return;
  const int i=t/n, j=t%n;
  const double dx=sites[2*i]-sites[2*j], dy=sites[2*i+1]-sites[2*j+1];
  double u=sqrt(dx*dx+dy*dy)/r, e;
  if (nu2==1) { e=exp(-u); value[t]=a*e; da[t]=e; dr[t]=a*e*u/r; }
  else if (nu2==3) { u*=sqrt(3.); e=exp(-u); value[t]=a*(1.+u)*e; da[t]=(1.+u)*e; dr[t]=a*u*u*e/r; }
  else { u*=sqrt(5.); e=exp(-u); value[t]=a*(1.+u+u*u/3.)*e; da[t]=(1.+u+u*u/3.)*e; dr[t]=a*e*(u*u+u*u*u)/(3.*r); }
}

void axpy(GpuContext& c, double alpha, const double* x, double* y, size_t count) {
  check_a(cublasDaxpy(c.cublas(), static_cast<int>(count), &alpha, x, 1, y, 1));
}
// Row-major left/right: out = left * right, where right is square m-by-m.
void right_product(GpuContext& c, const double* left, const double* right,
                   double* out, int n, int m) {
  const double one=1., zero=0.;
  check_a(cublasDgemm(c.cublas(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, m,
                      &one, right, m, left, m, &zero, out, m));
}
// Row-major output = left * right'.  cuBLAS sees its transpose = right * left'.
void product_transpose(GpuContext& c, const double* left, const double* right,
                       double* out, int n, int m) {
  const double one=1., zero=0.;
  check_a(cublasDgemm(c.cublas(), CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                      &one, right, m, left, m, &zero, out, n));
}
void triple_add(GpuContext& c, double* out, const double* left, const double* middle,
                const double* right, double scale) {
  const auto& r=c.region(); const size_t nn=size_t(r.n)*r.n;
  right_product(c,left,middle,r.b_scratch0,r.n,r.m);
  product_transpose(c,r.b_scratch0,right,r.a_scratch,r.n,r.m);
  axpy(c,scale,r.a_scratch,out,nn);
}
void first_a(GpuContext& c, double* out, const double* bi, const double* ki) {
  const auto& r=c.region();
  triple_add(c,out,bi,r.k,r.b,-1.);
  triple_add(c,out,r.b,ki,r.b,-1.);
  triple_add(c,out,r.b,r.k,bi,-1.);
}
void second_a(GpuContext& c, double* out, const double* bi, const double* bj,
              const double* bij, const double* ki, const double* kj,
              const double* kij) {
  const auto& r=c.region();
  triple_add(c,out,bij,r.k,r.b,-1.);
  triple_add(c,out,bi,kj,r.b,-1.);
  triple_add(c,out,bi,r.k,bj,-1.);
  triple_add(c,out,bj,ki,r.b,-1.);
  triple_add(c,out,r.b,kij,r.b,-1.);
  triple_add(c,out,r.b,ki,bj,-1.);
  triple_add(c,out,bj,r.k,bi,-1.);
  triple_add(c,out,r.b,kj,bi,-1.);
  triple_add(c,out,r.b,r.k,bij,-1.);
}
}

DeviceADerivativeSlice compute_device_a_derivatives(GpuContext& c, int nu2,
                                                     double alpha, double range) {
  if ((nu2 != 1 && nu2 != 3 && nu2 != 5) || alpha <= 0. || range <= 0.)
    throw std::runtime_error("invalid device A derivative parameters");
  check_a(cudaSetDevice(c.device_id_)); auto& r=c.region_;
  if (!r.initialized) throw std::runtime_error("device A derivatives require bound region");
  const auto b=compute_device_b_derivatives(c,nu2,alpha,range);
  const size_t nn=size_t(r.n)*r.n;
  if (!r.a_alpha) {
    for (double** p : {&r.a_value,&r.a_alpha,&r.a_range,&r.a_alpha_alpha,&r.a_alpha_range,
                       &r.a_range_alpha,&r.a_range_range,&r.a_scratch})
      check_a(cudaMalloc(p,nn*sizeof(double)));
    ++r.a_derivative_allocation_count;
  }
  const auto cjj=compute_device_cjj_second_derivatives(c,nu2,alpha,range);
  constexpr int threads=256;
  cjj_value_first_derivatives<<<(nn+threads-1)/threads,threads,0,c.stream_>>>(
      r.sites,r.n,nu2,alpha,range,r.a_value,r.a_alpha,r.a_range);
  check_a(cudaGetLastError());
  check_a(cudaMemcpyAsync(r.a_alpha_alpha,cjj.alpha_alpha,nn*sizeof(double),cudaMemcpyDeviceToDevice,c.stream_));
  check_a(cudaMemcpyAsync(r.a_alpha_range,cjj.alpha_range,nn*sizeof(double),cudaMemcpyDeviceToDevice,c.stream_));
  check_a(cudaMemcpyAsync(r.a_range_alpha,cjj.range_alpha,nn*sizeof(double),cudaMemcpyDeviceToDevice,c.stream_));
  check_a(cudaMemcpyAsync(r.a_range_range,cjj.range_range,nn*sizeof(double),cudaMemcpyDeviceToDevice,c.stream_));
  triple_add(c,r.a_value,r.b,r.k,r.b,-1.);
  first_a(c,r.a_alpha,b.alpha,r.k_alpha);
  first_a(c,r.a_range,b.range,r.k_range);
  second_a(c,r.a_alpha_alpha,b.alpha,b.alpha,b.alpha_alpha,r.k_alpha,r.k_alpha,r.k_alpha_alpha);
  second_a(c,r.a_alpha_range,b.alpha,b.range,b.alpha_range,r.k_alpha,r.k_range,r.k_alpha_range);
  second_a(c,r.a_range_alpha,b.range,b.alpha,b.range_alpha,r.k_range,r.k_alpha,r.k_alpha_range);
  second_a(c,r.a_range_range,b.range,b.range,b.range_range,r.k_range,r.k_range,r.k_range_range);
  return {r.n,r.a_value,r.a_alpha,r.a_range,r.a_alpha_alpha,r.a_alpha_range,r.a_range_alpha,r.a_range_range};
}

ASecondDerivatives download_device_a_second_derivatives(
    GpuContext& c, const DeviceADerivativeSlice& s) {
  if (!s.alpha_alpha || s.n <= 0) throw std::runtime_error("invalid device A derivative slice");
  check_a(cudaSetDevice(c.device_id_)); const size_t nn=size_t(s.n)*s.n;
  ASecondDerivatives out{Matrix(s.n,s.n),Matrix(s.n,s.n),Matrix(s.n,s.n),Matrix(s.n,s.n)};
  auto load=[&](Matrix& host,const double* device) { check_a(cudaMemcpyAsync(host.a.data(),device,nn*sizeof(double),cudaMemcpyDeviceToHost,c.stream_)); };
  load(out.aa,s.alpha_alpha); load(out.ar,s.alpha_range); load(out.ra,s.range_alpha); load(out.rr,s.range_range);
  check_a(cudaStreamSynchronize(c.stream_)); c.region_.result_d2h_bytes += static_cast<long long>(4*nn*sizeof(double));
  return out;
}

std::pair<AFirstDerivatives, ASecondDerivatives> download_device_a_derivatives(
    GpuContext& c, const DeviceADerivativeSlice& s) {
  if (!s.alpha || !s.alpha_alpha || s.n <= 0) throw std::runtime_error("invalid device A derivative slice");
  check_a(cudaSetDevice(c.device_id_)); const size_t nn=size_t(s.n)*s.n;
  AFirstDerivatives first{Matrix(s.n,s.n),Matrix(s.n,s.n)};
  ASecondDerivatives second{Matrix(s.n,s.n),Matrix(s.n,s.n),Matrix(s.n,s.n),Matrix(s.n,s.n)};
  auto load=[&](Matrix& host,const double* device) { check_a(cudaMemcpyAsync(host.a.data(),device,nn*sizeof(double),cudaMemcpyDeviceToHost,c.stream_)); };
  load(first.alpha,s.alpha); load(first.range,s.range); load(second.aa,s.alpha_alpha);
  load(second.ar,s.alpha_range); load(second.ra,s.range_alpha); load(second.rr,s.range_range);
  check_a(cudaStreamSynchronize(c.stream_)); c.region_.result_d2h_bytes += static_cast<long long>(6*nn*sizeof(double));
  return {std::move(first),std::move(second)};
}
} // namespace fedspatial
