#include "fedspatial/gpu_context.hpp"

#include <stdexcept>

namespace fedspatial {
namespace {
void check_device(cudaError_t value, const std::string& operation = "CUDA device B slice call") {
  if (value != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorName(value) +
                             " (" + cudaGetErrorString(value) + ")");
}
void check_device(cublasStatus_t value) { if (value != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS device B slice failure"); }
void check_device(cusolverStatus_t value) { if (value != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER device B slice failure"); }
__global__ void matern_value(const double* x, const double* y, int nx, int ny,
                             int nu2, double alpha, double range, double* out) {
  const int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nx*ny) return;
  const int i=t/ny,j=t%ny; double dx=x[2*i]-y[2*j],dy=x[2*i+1]-y[2*j+1];
  double u=sqrt(dx*dx+dy*dy)/range, poly;
  if(nu2==1) poly=exp(-u);
  else if(nu2==3) { u*=sqrt(3.); poly=(1.+u)*exp(-u); }
  else { u*=sqrt(5.); poly=(1.+u+u*u/3.)*exp(-u); }
  out[t]=alpha*poly;
}
__global__ void matern_derivatives(const double* x,const double* y,int nx,int ny,
                                   int nu2,double a,double r,double* da,double* dr,
                                   double* daa,double* dar,double* drr) {
  const int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nx*ny)return;
  const int i=t/ny,j=t%ny; double dx=x[2*i]-y[2*j],dy=x[2*i+1]-y[2*j+1];
  double u=sqrt(dx*dx+dy*dy)/r,e,p,rb,rbb;
  if(nu2==1){e=exp(-u);p=e;rb=e*u/r;rbb=a*e*(u*u-2*u)/(r*r);}
  else if(nu2==3){u*=sqrt(3.);e=exp(-u);p=(1.+u)*e;rb=e*u*u/r;rbb=a*e*(u*u*u-3.*u*u)/(r*r);}
  else {u*=sqrt(5.);e=exp(-u);p=(1.+u+u*u/3.)*e;rb=e*(u*u+u*u*u)/(3.*r);rbb=a*e*(u*u*u*u-3.*u*u*u-3.*u*u)/(3.*r*r);}
  da[t]=p;dr[t]=a*rb;daa[t]=0.;dar[t]=rb;drr[t]=rbb;
}
__global__ void matern_first_derivative(const double* x,const double* y,int nx,int ny,
                                        int nu2,double a,double r,int parameter,double* out) {
  const int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nx*ny)return;
  const int i=t/ny,j=t%ny; double dx=x[2*i]-y[2*j],dy=x[2*i+1]-y[2*j+1];
  double u=sqrt(dx*dx+dy*dy)/r,e,p,rb;
  if(nu2==1){e=exp(-u);p=e;rb=e*u/r;}
  else if(nu2==3){u*=sqrt(3.);e=exp(-u);p=(1.+u)*e;rb=e*u*u/r;}
  else {u*=sqrt(5.);e=exp(-u);p=(1.+u+u*u/3.)*e;rb=e*(u*u+u*u*u)/(3.*r);}
  out[t]=parameter==0?p:a*rb;
}
__global__ void matern_second_derivative(const double* x,const double* y,int nx,int ny,
                                         int nu2,double a,double r,int i,int j,double* out) {
  const int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nx*ny)return;
  const int row=t/ny,col=t%ny; double dx=x[2*row]-y[2*col],dy=x[2*row+1]-y[2*col+1];
  double u=sqrt(dx*dx+dy*dy)/r,e,ar,rr;
  if(nu2==1){e=exp(-u);ar=e*u/r;rr=a*e*(u*u-2.*u)/(r*r);}
  else if(nu2==3){u*=sqrt(3.);e=exp(-u);ar=e*u*u/r;rr=a*e*(u*u*u-3.*u*u)/(r*r);}
  else {u*=sqrt(5.);e=exp(-u);ar=e*(u*u+u*u*u)/(3.*r);rr=a*e*(u*u*u*u-3.*u*u*u-3.*u*u)/(3.*r*r);}
  out[t]=(i==0&&j==0)?0.:((i==1&&j==1)?rr:ar);
}
void right_solve(GpuContext& c,const double* in,double* out,int n,int m) {
  const double one=1.; const auto& r=c.region(); if (in!=out) check_device(cudaMemcpyAsync(out,in,size_t(n)*m*sizeof(double),cudaMemcpyDeviceToDevice,c.stream()));
  check_device(cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,m,n,&one,r.chol_k,m,out,m));
  check_device(cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,m,n,&one,r.chol_k,m,out,m));
}
void right_product(GpuContext& c,const double* left,const double* right,double* out,int n,int m) {
  const double one=1.,zero=0.;
  // row-major left/right are column-major left' and symmetric right.
  check_device(cublasDgemm(c.cublas(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,m,&one,right,m,left,m,&zero,out,m));
}
void axpy_device(GpuContext& c,double scale,const double* x,double* y,size_t n) { check_device(cublasDaxpy(c.cublas(),static_cast<int>(n),&scale,x,1,y,1)); }
DeviceBSlice cached_or_compute_base(GpuContext& c,int nu2,double alpha,double range) {
  const auto& r=c.region();
  if (r.producer_state_cache_active && r.producer_state_cache_valid) {
    ++const_cast<RegionGpuState&>(r).producer_b_base_reuse_count;
    return {r.n,r.m,r.cnk,r.k,r.chol_k,r.b};
  }
  return compute_device_b_slice(c,nu2,alpha,range);
}
void second_b(GpuContext& c,const double* cij,const double* ci,const double* kj,const double* bj,
              const double* ki,const double* kij,double* out) {
  const auto& r=c.region(); const int n=r.n,m=r.m; const size_t nm=size_t(n)*m;
  right_solve(c,cij,out,n,m);
  right_solve(c,ci,r.b_scratch0,n,m); right_product(c,r.b_scratch0,kj,r.b_scratch1,n,m); right_solve(c,r.b_scratch1,r.b_scratch0,n,m); axpy_device(c,-1.,r.b_scratch0,out,nm);
  right_product(c,bj,ki,r.b_scratch1,n,m); right_solve(c,r.b_scratch1,r.b_scratch0,n,m); axpy_device(c,-1.,r.b_scratch0,out,nm);
  right_product(c,r.b,kij,r.b_scratch1,n,m); right_solve(c,r.b_scratch1,r.b_scratch0,n,m); axpy_device(c,-1.,r.b_scratch0,out,nm);
  right_product(c,r.b,ki,r.b_scratch1,n,m); right_solve(c,r.b_scratch1,r.b_scratch0,n,m); right_product(c,r.b_scratch0,kj,r.b_scratch1,n,m); right_solve(c,r.b_scratch1,r.b_scratch0,n,m); axpy_device(c,1.,r.b_scratch0,out,nm);
}
}

DeviceBSlice compute_device_b_slice(GpuContext& context, int nu2, double alpha,
                                    double range) {
  if ((nu2!=1 && nu2!=3 && nu2!=5) || alpha<=0. || range<=0.) throw std::runtime_error("invalid device B slice parameters");
  check_device(cudaSetDevice(context.device_id_), "cudaSetDevice device-B-slice"); auto& r=context.region_;
  if (!r.initialized) throw std::runtime_error("device B slice requires bound region");
  const int threads=256; const double one=1.;
  matern_value<<<(r.n*r.m+threads-1)/threads,threads,0,context.stream_>>>(r.sites,r.knots,r.n,r.m,nu2,alpha,range,r.cnk);
  matern_value<<<(r.m*r.m+threads-1)/threads,threads,0,context.stream_>>>(r.knots,r.knots,r.m,r.m,nu2,alpha,range,r.k);
  check_device(cudaGetLastError(), "matern_value launches device-B-slice nx=" + std::to_string(r.n) + " m=" + std::to_string(r.m)); check_device(cudaMemcpyAsync(r.chol_k,r.k,size_t(r.m)*r.m*sizeof(double),cudaMemcpyDeviceToDevice,context.stream_), "cudaMemcpyAsync K->chol_K D2D bytes=" + std::to_string(size_t(r.m)*r.m*sizeof(double)) + " src=" + std::to_string(reinterpret_cast<uintptr_t>(r.k)) + " dst=" + std::to_string(reinterpret_cast<uintptr_t>(r.chol_k)));
  int work=0; check_device(cusolverDnDpotrf_bufferSize(context.cusolver_,CUBLAS_FILL_MODE_LOWER,r.m,r.chol_k,r.m,&work));
  if (r.potrf_work_capacity < static_cast<size_t>(work)) { if(r.potrf_work) check_device(cudaFree(r.potrf_work), "cudaFree old potrf workspace"); check_device(cudaMalloc(&r.potrf_work,size_t(work)*sizeof(double)), "cudaMalloc potrf workspace elements=" + std::to_string(work)); r.potrf_work_capacity=work; }
  check_device(cusolverDnDpotrf(context.cusolver_,CUBLAS_FILL_MODE_LOWER,r.m,r.chol_k,r.m,r.potrf_work,work,r.potrf_info));
  int info=-1; check_device(cudaMemcpyAsync(&info,r.potrf_info,sizeof(int),cudaMemcpyDeviceToHost,context.stream_), "cudaMemcpyAsync POTRF info D2H"); check_device(cudaStreamSynchronize(context.stream_), "cudaStreamSynchronize after K POTRF n=" + std::to_string(r.m)); if(info) throw std::runtime_error("device B slice K is not SPD");
  // Row-major C is interpreted as column-major C^T (m by n): B^T=K^-1 C^T.
  check_device(cudaMemcpyAsync(r.b,r.cnk,size_t(r.n)*r.m*sizeof(double),cudaMemcpyDeviceToDevice,context.stream_), "cudaMemcpyAsync Cnk->B D2D bytes=" + std::to_string(size_t(r.n)*r.m*sizeof(double)) + " src=" + std::to_string(reinterpret_cast<uintptr_t>(r.cnk)) + " dst=" + std::to_string(reinterpret_cast<uintptr_t>(r.b)));
  check_device(cublasDtrsm(context.cublas_,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,r.m,r.n,&one,r.chol_k,r.m,r.b,r.m));
  check_device(cublasDtrsm(context.cublas_,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,r.m,r.n,&one,r.chol_k,r.m,r.b,r.m));
  return {r.n,r.m,r.cnk,r.k,r.chol_k,r.b};
}

DeviceBFirstDerivativeSlice compute_device_b_first_derivative(
    GpuContext& context, int nu2, double alpha, double range, int parameter) {
  if ((nu2!=1 && nu2!=3 && nu2!=5) || alpha<=0. || range<=0. ||
      (parameter!=0 && parameter!=1))
    throw std::runtime_error("invalid first-order device B derivative parameters");
  const auto base=cached_or_compute_base(context,nu2,alpha,range);
  check_device(cudaSetDevice(context.device_id_)); auto& r=context.region_;
  const size_t nm=size_t(r.n)*r.m, mm=size_t(r.m)*r.m;
  if (!r.b_first) {
    check_device(cudaMalloc(&r.b_first,nm*sizeof(double)));
    check_device(cudaMalloc(&r.k_first,mm*sizeof(double)));
    ++r.b_first_derivative_allocation_count;
  }
  // Phase A: derivative covariance values overwrite the sole n-by-m output.
  // Phase B: it is transformed in place to K^-1 C_i.  Phase C borrows bk
  // only for B K_i K^-1, after no local-state consumer needs bk's old value.
  constexpr int threads=256;
  matern_first_derivative<<<(r.n*r.m+threads-1)/threads,threads,0,context.stream_>>>(
      r.sites,r.knots,r.n,r.m,nu2,alpha,range,parameter,r.b_first);
  matern_first_derivative<<<(r.m*r.m+threads-1)/threads,threads,0,context.stream_>>>(
      r.knots,r.knots,r.m,r.m,nu2,alpha,range,parameter,r.k_first);
  check_device(cudaGetLastError());
  check_device(cudaMemcpyAsync(r.bk,r.b_first,nm*sizeof(double),cudaMemcpyDeviceToDevice,context.stream_));
  right_solve(context,r.bk,r.bk,r.n,r.m);
  check_device(cudaMemcpyAsync(r.b_first,r.bk,nm*sizeof(double),cudaMemcpyDeviceToDevice,context.stream_));
  right_product(context,base.b,r.k_first,r.bk,r.n,r.m);
  right_solve(context,r.bk,r.bk,r.n,r.m);
  axpy_device(context,-1.,r.bk,r.b_first,nm);
  return {r.n,r.m,r.b_first};
}

DeviceBPairDerivativeSlice compute_device_b_pair_derivatives(
    GpuContext& context,int nu2,double alpha,double range,int i,int j) {
  if ((i!=0 && i!=1) || (j!=0 && j!=1))
    throw std::runtime_error("invalid pairwise device B derivative parameters");
  const auto base=cached_or_compute_base(context,nu2,alpha,range);
  check_device(cudaSetDevice(context.device_id_)); auto& r=context.region_;
  const size_t nm=size_t(r.n)*r.m, mm=size_t(r.m)*r.m;
  if(!r.b_pair_first) {
    if(!r.b_first) check_device(cudaMalloc(&r.b_first,nm*sizeof(double)));
    if(!r.k_first) check_device(cudaMalloc(&r.k_first,mm*sizeof(double)));
    for(double** p:{&r.b_pair_first,&r.b_pair_second}) check_device(cudaMalloc(p,nm*sizeof(double)));
    for(double** p:{&r.k_pair,&r.k_second}) check_device(cudaMalloc(p,mm*sizeof(double)));
    // b_first/b_pair_first/b_pair_second and their K counterparts are the
    // exact complete-pair cache.  Keep the capacities after release for the
    // synchronous allocation audit.
    r.b_pair_nm_capacity=nm;
    r.b_pair_mm_capacity=mm;
    ++r.b_pair_derivative_allocation_count;
  }
  constexpr int threads=256;
  auto first=[&](int p,double* out,double* kp) {
    matern_first_derivative<<<(r.n*r.m+threads-1)/threads,threads,0,context.stream_>>>(r.sites,r.knots,r.n,r.m,nu2,alpha,range,p,out);
    matern_first_derivative<<<(r.m*r.m+threads-1)/threads,threads,0,context.stream_>>>(r.knots,r.knots,r.m,r.m,nu2,alpha,range,p,kp);
    check_device(cudaGetLastError());
    check_device(cudaMemcpyAsync(r.bk,out,nm*sizeof(double),cudaMemcpyDeviceToDevice,context.stream_));
    right_solve(context,r.bk,out,r.n,r.m);
    right_product(context,base.b,kp,r.bk,r.n,r.m); right_solve(context,r.bk,r.bk,r.n,r.m);
    axpy_device(context,-1.,r.bk,out,nm);
  };
  first(i,r.b_first,r.k_first);
  const double* bj=r.b_first; const double* kj=r.k_first;
  if(i!=j) { first(j,r.b_pair_first,r.k_pair); bj=r.b_pair_first; kj=r.k_pair; }
  matern_second_derivative<<<(r.n*r.m+threads-1)/threads,threads,0,context.stream_>>>(r.sites,r.knots,r.n,r.m,nu2,alpha,range,i,j,r.b_pair_second);
  matern_second_derivative<<<(r.m*r.m+threads-1)/threads,threads,0,context.stream_>>>(r.knots,r.knots,r.m,r.m,nu2,alpha,range,i,j,r.k_second);
  check_device(cudaGetLastError());
  check_device(cudaMemcpyAsync(r.bk,r.b_pair_second,nm*sizeof(double),cudaMemcpyDeviceToDevice,context.stream_));
  right_solve(context,r.bk,r.b_pair_second,r.n,r.m);
  // Rebuild C_i in the base Cnk buffer.  It is scratch after B is fixed.
  matern_first_derivative<<<(r.n*r.m+threads-1)/threads,threads,0,context.stream_>>>(r.sites,r.knots,r.n,r.m,nu2,alpha,range,i,r.cnk);
  right_solve(context,r.cnk,r.cnk,r.n,r.m); right_product(context,r.cnk,kj,r.bk,r.n,r.m); right_solve(context,r.bk,r.bk,r.n,r.m); axpy_device(context,-1.,r.bk,r.b_pair_second,nm);
  right_product(context,bj,r.k_first,r.bk,r.n,r.m); right_solve(context,r.bk,r.bk,r.n,r.m); axpy_device(context,-1.,r.bk,r.b_pair_second,nm);
  right_product(context,base.b,r.k_second,r.bk,r.n,r.m); right_solve(context,r.bk,r.bk,r.n,r.m); axpy_device(context,-1.,r.bk,r.b_pair_second,nm);
  right_product(context,base.b,r.k_first,r.cnk,r.n,r.m); right_solve(context,r.cnk,r.cnk,r.n,r.m); right_product(context,r.cnk,kj,r.bk,r.n,r.m); right_solve(context,r.bk,r.bk,r.n,r.m); axpy_device(context,1.,r.bk,r.b_pair_second,nm);
  return {r.n,r.m,r.b_first,bj,r.b_pair_second,r.k_first,kj,r.k_second};
}

void release_device_delta_theta_pair_workspace(GpuContext& context) {
  check_device(cudaSetDevice(context.device_id_));
  auto& r = context.region_;
  // These are only a completed local theta/theta contraction cache.  The
  // following central global-K stage never reads them.
  for (double** p : {&r.b_first, &r.b_pair_first, &r.b_pair_second,
                     &r.k_first, &r.k_pair, &r.k_second}) {
    if (*p) { check_device(cudaFree(*p)); *p = nullptr; }
  }
}

Matrix download_device_b_slice(GpuContext& context, const DeviceBSlice& slice) {
  if (slice.n<=0 || slice.m<=0 || !slice.b) throw std::runtime_error("invalid device B slice download");
  check_device(cudaSetDevice(context.device_id_)); Matrix out(slice.n,slice.m);
  const size_t bytes=size_t(slice.n)*slice.m*sizeof(double);
  check_device(cudaMemcpyAsync(out.a.data(),slice.b,bytes,cudaMemcpyDeviceToHost,context.stream_));
  check_device(cudaStreamSynchronize(context.stream_));
  context.region_.result_d2h_bytes+=static_cast<long long>(bytes);
  return out;
}

DeviceBDerivativeSlice compute_device_b_derivatives(GpuContext& context,int nu2,double alpha,double range) {
  const auto base=cached_or_compute_base(context,nu2,alpha,range); check_device(cudaSetDevice(context.device_id_));
  auto& r=context.region_;
  const bool allocating=!r.b_alpha;
  if(allocating) { const size_t nm=size_t(r.n)*r.m,mm=size_t(r.m)*r.m; auto alloc=[&](double** p,size_t count){check_device(cudaMalloc(p,count*sizeof(double)));};
    for(double** p:{&r.cnk_alpha,&r.cnk_range,&r.cnk_alpha_alpha,&r.cnk_alpha_range,&r.cnk_range_range,&r.b_alpha,&r.b_range,&r.b_alpha_alpha,&r.b_alpha_range,&r.b_range_alpha,&r.b_range_range,&r.b_scratch0,&r.b_scratch1}) alloc(p,nm);
    for(double** p:{&r.k_alpha,&r.k_range,&r.k_alpha_alpha,&r.k_alpha_range,&r.k_range_range}) alloc(p,mm); }
  if (allocating) ++r.b_derivative_allocation_count;
  const int threads=256;
  matern_derivatives<<<(r.n*r.m+threads-1)/threads,threads,0,context.stream_>>>(r.sites,r.knots,r.n,r.m,nu2,alpha,range,r.cnk_alpha,r.cnk_range,r.cnk_alpha_alpha,r.cnk_alpha_range,r.cnk_range_range);
  matern_derivatives<<<(r.m*r.m+threads-1)/threads,threads,0,context.stream_>>>(r.knots,r.knots,r.m,r.m,nu2,alpha,range,r.k_alpha,r.k_range,r.k_alpha_alpha,r.k_alpha_range,r.k_range_range);
  check_device(cudaGetLastError()); const size_t nm=size_t(r.n)*r.m;
  right_solve(context,r.cnk_alpha,r.b_alpha,r.n,r.m); right_product(context,r.b,r.k_alpha,r.b_scratch1,r.n,r.m); right_solve(context,r.b_scratch1,r.b_scratch0,r.n,r.m); axpy_device(context,-1.,r.b_scratch0,r.b_alpha,nm);
  right_solve(context,r.cnk_range,r.b_range,r.n,r.m); right_product(context,r.b,r.k_range,r.b_scratch1,r.n,r.m); right_solve(context,r.b_scratch1,r.b_scratch0,r.n,r.m); axpy_device(context,-1.,r.b_scratch0,r.b_range,nm);
  second_b(context,r.cnk_alpha_alpha,r.cnk_alpha,r.k_alpha,r.b_alpha,r.k_alpha,r.k_alpha_alpha,r.b_alpha_alpha);
  second_b(context,r.cnk_alpha_range,r.cnk_alpha,r.k_range,r.b_range,r.k_alpha,r.k_alpha_range,r.b_alpha_range);
  second_b(context,r.cnk_alpha_range,r.cnk_range,r.k_alpha,r.b_alpha,r.k_range,r.k_alpha_range,r.b_range_alpha);
  second_b(context,r.cnk_range_range,r.cnk_range,r.k_range,r.b_range,r.k_range,r.k_range_range,r.b_range_range);
  return {r.n,r.m,r.b_alpha,r.b_range,r.b_alpha_alpha,r.b_alpha_range,r.b_range_alpha,r.b_range_range};
}

BSecondDerivatives download_device_b_derivatives(GpuContext& context,const DeviceBDerivativeSlice& s) {
  if(!s.alpha||s.n<=0||s.m<=0)throw std::runtime_error("invalid device B derivative slice"); check_device(cudaSetDevice(context.device_id_));
  BSecondDerivatives out; auto load=[&](Matrix& h,const double* d){h=Matrix(s.n,s.m);check_device(cudaMemcpyAsync(h.a.data(),d,size_t(s.n)*s.m*sizeof(double),cudaMemcpyDeviceToHost,context.stream_));};
  load(out.baa,s.alpha_alpha);load(out.bar,s.alpha_range);load(out.bra,s.range_alpha);load(out.brr,s.range_range);check_device(cudaStreamSynchronize(context.stream_));context.region_.result_d2h_bytes+=static_cast<long long>(4*size_t(s.n)*s.m*sizeof(double));return out;
}
}  // namespace fedspatial
