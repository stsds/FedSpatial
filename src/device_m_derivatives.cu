#include "fedspatial/gpu_context.hpp"
#include <stdexcept>

namespace fedspatial {
namespace {
void check_m(cudaError_t value) { if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value)); }
void check_m(cublasStatus_t value) { if (value != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS M derivative failure"); }
__global__ void zero_matrix(double* value, size_t count) { const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(i<count)value[i]=0.; }
void copy(GpuContext& c,double* out,const double* in,size_t count) { check_m(cudaMemcpyAsync(out,in,count*sizeof(double),cudaMemcpyDeviceToDevice,c.stream())); }
void scale(GpuContext& c,double* value,double factor,size_t count) { check_m(cublasDscal(c.cublas(),static_cast<int>(count),&factor,value,1)); }
}

DeviceMDerivativeSlice compute_device_m_derivatives(GpuContext& c, int nu2,
                                                     double delta, double alpha, double range) {
  if (delta <= 0. || alpha <= 0. || range <= 0.) throw std::runtime_error("invalid device M derivative parameters");
  check_m(cudaSetDevice(c.device_id_)); auto& r=c.region_;
  if (!r.initialized) throw std::runtime_error("device M derivatives require bound region");
  const auto a=compute_device_a_derivatives(c,nu2,alpha,range);
  const size_t nn=size_t(r.n)*r.n;
  if (!r.m_delta) {
    for(double** p:{&r.m_delta,&r.m_alpha,&r.m_range,&r.m_dd,&r.m_da,&r.m_dr,&r.m_aa,&r.m_ar,&r.m_ra,&r.m_rr})
      check_m(cudaMalloc(p,nn*sizeof(double)));
    ++r.m_derivative_allocation_count;
  }
  // Exact legacy ordering: M_delta=A; M_alpha/range=delta*A_i; the mixed
  // delta derivatives divide those stored first derivatives back by delta.
  copy(c,r.m_delta,a.value,nn);
  copy(c,r.m_alpha,a.alpha,nn); scale(c,r.m_alpha,delta,nn);
  copy(c,r.m_range,a.range,nn); scale(c,r.m_range,delta,nn);
  zero_matrix<<<(nn+255)/256,256,0,c.stream_>>>(r.m_dd,nn); check_m(cudaGetLastError());
  copy(c,r.m_da,r.m_alpha,nn); scale(c,r.m_da,1./delta,nn);
  copy(c,r.m_dr,r.m_range,nn); scale(c,r.m_dr,1./delta,nn);
  copy(c,r.m_aa,a.alpha_alpha,nn); scale(c,r.m_aa,delta,nn);
  copy(c,r.m_ar,a.alpha_range,nn); scale(c,r.m_ar,delta,nn);
  copy(c,r.m_ra,a.range_alpha,nn); scale(c,r.m_ra,delta,nn);
  copy(c,r.m_rr,a.range_range,nn); scale(c,r.m_rr,delta,nn);
  return {r.n,r.m_delta,r.m_alpha,r.m_range,r.m_dd,r.m_da,r.m_dr,r.m_da,r.m_aa,r.m_ar,r.m_dr,r.m_ra,r.m_rr};
}

std::pair<MFirstDerivatives,MSecondDerivatives> download_device_m_derivatives(
    GpuContext& c,const DeviceMDerivativeSlice& s) {
  if(!s.delta || !s.dd || s.n<=0) throw std::runtime_error("invalid device M derivative slice");
  check_m(cudaSetDevice(c.device_id_)); const size_t nn=size_t(s.n)*s.n;
  MFirstDerivatives first{Matrix(s.n,s.n),Matrix(s.n,s.n),Matrix(s.n,s.n)};
  MSecondDerivatives second{Matrix(s.n,s.n),Matrix(s.n,s.n),Matrix(s.n,s.n),Matrix(s.n,s.n),Matrix(s.n,s.n),Matrix(s.n,s.n),Matrix(s.n,s.n),Matrix(s.n,s.n),Matrix(s.n,s.n)};
  auto load=[&](Matrix& host,const double* device){check_m(cudaMemcpyAsync(host.a.data(),device,nn*sizeof(double),cudaMemcpyDeviceToHost,c.stream()));};
  load(first.delta,s.delta);load(first.alpha,s.alpha);load(first.range,s.range);load(second.dd,s.dd);load(second.da,s.da);load(second.dr,s.dr);load(second.ad,s.ad);load(second.aa,s.aa);load(second.ar,s.ar);load(second.rd,s.rd);load(second.ra,s.ra);load(second.rr,s.rr);
  check_m(cudaStreamSynchronize(c.stream_)); c.region_.result_d2h_bytes+=static_cast<long long>(12*nn*sizeof(double));
  return {std::move(first),std::move(second)};
}
} // namespace fedspatial
