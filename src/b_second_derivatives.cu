#include "fedspatial/gpu_context.hpp"
#include <stdexcept>
namespace fedspatial { namespace {
__global__ void matern_second(const double*x,const double*y,double*aa,double*ar,double*rr,int nx,int ny,int nu2,double alpha,double range){int t=blockIdx.x*blockDim.x+threadIdx.x;if(t<nx*ny){int i=t/ny,j=t%ny;double dx=x[2*i]-y[2*j],dy=x[2*i+1]-y[2*j+1],u=sqrt(dx*dx+dy*dy)/range,e;if(nu2==1){e=exp(-u);aa[t]=0;ar[t]=e*u/range;rr[t]=alpha*e*(u*u-2*u)/(range*range);}else if(nu2==3){u*=sqrt(3.);e=exp(-u);aa[t]=0;ar[t]=e*u*u/range;rr[t]=alpha*e*(u*u*u-3*u*u)/(range*range);}else{u*=sqrt(5.);e=exp(-u);aa[t]=0;ar[t]=e*(u*u+u*u*u)/(3*range);rr[t]=alpha*e*(u*u*u*u-3*u*u*u-3*u*u)/(3*range*range);}}}
__global__ void matern_first2(const double*x,const double*y,double*da,double*dr,int nx,int ny,int nu2,double alpha,double range){int t=blockIdx.x*blockDim.x+threadIdx.x;if(t<nx*ny){int i=t/ny,j=t%ny;double dx=x[2*i]-y[2*j],dy=x[2*i+1]-y[2*j+1],u=sqrt(dx*dx+dy*dy)/range,e;if(nu2==1){e=exp(-u);da[t]=e;dr[t]=alpha*e*u/range;}else if(nu2==3){u*=sqrt(3.);e=exp(-u);da[t]=(1+u)*e;dr[t]=alpha*u*u*e/range;}else{u*=sqrt(5.);e=exp(-u);da[t]=(1+u+u*u/3)*e;dr[t]=alpha*e*(u*u+u*u*u)/(3*range);}}}
void ck2(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}Matrix rs(const Matrix&a,const Matrix&k){return transpose(solve_spd(k,transpose(a)));}Matrix rs_profiled(GpuContext&c,const Matrix&a,const Matrix&k){PProfileScope scope(c.profile(),"b_right_solve");c.profile().add("b_trsm",0);return rs(a,k);}void add2(Matrix&a,const Matrix&b,double s=1){for(int i=0;i<a.rows*a.cols;i++)a.a[i]+=s*b.a[i];}
#define rs(a,k) rs_profiled(c,a,k)
Matrix cov2(GpuContext&c,const Matrix&x,const Matrix&y,int nu2,double a,double r,int which){Matrix o(x.rows,y.rows);double *dx,*dy,*aa,*ar,*rr;ck2(cudaMalloc(&dx,x.a.size()*sizeof(double)));ck2(cudaMalloc(&dy,y.a.size()*sizeof(double)));ck2(cudaMalloc(&aa,o.a.size()*sizeof(double)));ck2(cudaMalloc(&ar,o.a.size()*sizeof(double)));ck2(cudaMalloc(&rr,o.a.size()*sizeof(double)));ck2(cudaMemcpyAsync(dx,x.a.data(),x.a.size()*sizeof(double),cudaMemcpyHostToDevice,c.stream()));ck2(cudaMemcpyAsync(dy,y.a.data(),y.a.size()*sizeof(double),cudaMemcpyHostToDevice,c.stream()));matern_second<<<(o.a.size()+63)/64,64,0,c.stream()>>>(dx,dy,aa,ar,rr,x.rows,y.rows,nu2,a,r);ck2(cudaGetLastError());double*src=which==0?aa:which==1?ar:rr;ck2(cudaMemcpyAsync(o.a.data(),src,o.a.size()*sizeof(double),cudaMemcpyDeviceToHost,c.stream()));ck2(cudaStreamSynchronize(c.stream()));cudaFree(dx);cudaFree(dy);cudaFree(aa);cudaFree(ar);cudaFree(rr);return o;}
Matrix first(GpuContext&c,const Matrix&x,const Matrix&y,int nu2,double a,double r,bool rg){Matrix o(x.rows,y.rows);double *dx,*dy,*da,*dr;ck2(cudaMalloc(&dx,x.a.size()*sizeof(double)));ck2(cudaMalloc(&dy,y.a.size()*sizeof(double)));ck2(cudaMalloc(&da,o.a.size()*sizeof(double)));ck2(cudaMalloc(&dr,o.a.size()*sizeof(double)));ck2(cudaMemcpyAsync(dx,x.a.data(),x.a.size()*sizeof(double),cudaMemcpyHostToDevice,c.stream()));ck2(cudaMemcpyAsync(dy,y.a.data(),y.a.size()*sizeof(double),cudaMemcpyHostToDevice,c.stream()));matern_first2<<<(o.a.size()+63)/64,64,0,c.stream()>>>(dx,dy,da,dr,x.rows,y.rows,nu2,a,r);ck2(cudaGetLastError());ck2(cudaMemcpyAsync(o.a.data(),rg?dr:da,o.a.size()*sizeof(double),cudaMemcpyDeviceToHost,c.stream()));ck2(cudaStreamSynchronize(c.stream()));cudaFree(dx);cudaFree(dy);cudaFree(da);cudaFree(dr);return o;}
}
DeviceCjjSecondDerivativeSlice compute_device_cjj_second_derivatives(
    GpuContext& c, int nu2, double alpha, double range) {
  if ((nu2 != 1 && nu2 != 3 && nu2 != 5) || alpha <= 0. || range <= 0.)
    throw std::runtime_error("invalid device Cjj derivative parameters");
  ck2(cudaSetDevice(c.device_id_)); auto& region=c.region_;
  if (!region.initialized) throw std::runtime_error("device Cjj derivatives require bound region");
  const size_t nn=size_t(region.n)*region.n;
  if (!region.cjj_alpha_alpha) {
    ck2(cudaMalloc(&region.cjj_alpha_alpha,nn*sizeof(double)));
    ck2(cudaMalloc(&region.cjj_alpha_range,nn*sizeof(double)));
    ck2(cudaMalloc(&region.cjj_range_range,nn*sizeof(double)));
    ++region.cjj_second_allocation_count;
  }
  matern_second<<<(nn+255)/256,256,0,c.stream_>>>(
      region.sites,region.sites,region.cjj_alpha_alpha,region.cjj_alpha_range,
      region.cjj_range_range,region.n,region.n,nu2,alpha,range);
  ck2(cudaGetLastError());
  return {region.n,region.cjj_alpha_alpha,region.cjj_alpha_range,
          region.cjj_alpha_range,region.cjj_range_range};
}

CjjSecondDerivatives download_device_cjj_second_derivatives(
    GpuContext& c, const DeviceCjjSecondDerivativeSlice& slice) {
  if (!slice.alpha_alpha || slice.n <= 0) throw std::runtime_error("invalid device Cjj derivative slice");
  ck2(cudaSetDevice(c.device_id_)); const size_t nn=size_t(slice.n)*slice.n;
  Matrix aa(slice.n,slice.n),ar(slice.n,slice.n),rr(slice.n,slice.n);
  ck2(cudaMemcpyAsync(aa.a.data(),slice.alpha_alpha,nn*sizeof(double),cudaMemcpyDeviceToHost,c.stream_));
  ck2(cudaMemcpyAsync(ar.a.data(),slice.alpha_range,nn*sizeof(double),cudaMemcpyDeviceToHost,c.stream_));
  ck2(cudaMemcpyAsync(rr.a.data(),slice.range_range,nn*sizeof(double),cudaMemcpyDeviceToHost,c.stream_));
  ck2(cudaStreamSynchronize(c.stream_)); c.region_.result_d2h_bytes+=static_cast<long long>(3*nn*sizeof(double));
  return {aa,ar,ar,rr};
}

CjjSecondDerivatives compute_cjj_second_derivatives(GpuContext& c,const WorkerInput& w,const GlobalParameters& p){
  PProfileScope profile_scope(c.profile(), "compute_cjj_second_derivatives");
  // The active local/A path has already bound this worker. Keep this host
  // compatibility wrapper only for the remaining host A/M arithmetic.
  const auto& region=c.region();
  if (!region.initialized || region.n != w.sites.rows)
    throw std::runtime_error("compute_cjj_second_derivatives requires bound worker region");
  return download_device_cjj_second_derivatives(
      c,compute_device_cjj_second_derivatives(c,p.nu2,p.alpha,p.range));
}
PGlobalDerivativeWorkspace make_p_global_derivative_workspace(GpuContext&c,const Matrix&k,const GlobalParameters&p){c.profile().add("global_k",0);c.profile().add("global_k_alpha",0);c.profile().add("global_k_range",0);c.profile().add("global_k_alpha_alpha",0);c.profile().add("global_k_alpha_range",0);c.profile().add("global_k_range_range",0);const auto d=matern_with_derivatives(c,k,k,p.nu2,p.alpha,p.range);return {d.value,d.alpha,d.range,d.aa,d.ar,d.rr,cholesky_lower(d.value)};}
BSecondDerivatives compute_b_second_derivatives(GpuContext&c,const WorkerInput&w,const Matrix&knots,const GlobalParameters&p,const PGlobalDerivativeWorkspace&g){PProfileScope profile_scope(c.profile(), "compute_b_second_derivatives");auto local=compute_mu_sigma_local(c,w,knots,p);const Matrix&K=g.k;Matrix Ca=first(c,w.sites,knots,p.nu2,p.alpha,p.range,false),Cr=first(c,w.sites,knots,p.nu2,p.alpha,p.range,true),Ba=rs(Ca,K);add2(Ba,rs(matmul(local.B,g.k_alpha),K),-1);Matrix Br=rs(Cr,K);add2(Br,rs(matmul(local.B,g.k_range),K),-1);auto b2=[&](const Matrix&Cij,const Matrix&Ci,const Matrix&Kj,const Matrix&Bj,const Matrix&Ki,const Matrix&Kij){Matrix o=rs(Cij,K);add2(o,rs(matmul(rs(Ci,K),Kj),K),-1);add2(o,rs(matmul(Bj,Ki),K),-1);add2(o,rs(matmul(local.B,Kij),K),-1);add2(o,rs(matmul(rs(matmul(local.B,Ki),K),Kj),K));return o;};return {g.k_alpha_alpha,g.k_alpha_range,g.k_range_range,cov2(c,w.sites,knots,p.nu2,p.alpha,p.range,0),cov2(c,w.sites,knots,p.nu2,p.alpha,p.range,1),cov2(c,w.sites,knots,p.nu2,p.alpha,p.range,2),b2(cov2(c,w.sites,knots,p.nu2,p.alpha,p.range,0),Ca,g.k_alpha,Ba,g.k_alpha,g.k_alpha_alpha),b2(cov2(c,w.sites,knots,p.nu2,p.alpha,p.range,1),Ca,g.k_range,Br,g.k_alpha,g.k_alpha_range),b2(cov2(c,w.sites,knots,p.nu2,p.alpha,p.range,1),Cr,g.k_alpha,Ba,g.k_range,g.k_alpha_range),b2(cov2(c,w.sites,knots,p.nu2,p.alpha,p.range,2),Cr,g.k_range,Br,g.k_range,g.k_range_range)};}
CnkDerivativeWorkspace make_cnk_derivative_workspace(GpuContext&c,const WorkerInput&w,const Matrix&k,const GlobalParameters&p){c.profile().add("cnk_fused",0);return compute_cnk_derivatives_fused(c,w.sites,k,p.nu2,p.alpha,p.range);}
BSecondDerivatives compute_b_second_derivatives(GpuContext&c,const WorkerInput&w,const Matrix&knots,const GlobalParameters&p,const PGlobalDerivativeWorkspace&g,const CnkDerivativeWorkspace&d){PProfileScope profile_scope(c.profile(), "compute_b_second_derivatives");auto local=compute_mu_sigma_local(c,w,knots,p);const Matrix&Ca=d.alpha;const Matrix&Cr=d.range;auto solve=[&](const Matrix&x){return gpu_right_spd_solve(c,x,g.chol_k);};Matrix Ba=solve(Ca);add2(Ba,solve(matmul(local.B,g.k_alpha)),-1);Matrix Br=solve(Cr);add2(Br,solve(matmul(local.B,g.k_range)),-1);auto b2=[&](const Matrix&Cij,const Matrix&Ci,const Matrix&Kj,const Matrix&Bj,const Matrix&Ki,const Matrix&Kij){Matrix o=solve(Cij);add2(o,solve(matmul(solve(Ci),Kj)),-1);add2(o,solve(matmul(Bj,Ki)),-1);add2(o,solve(matmul(local.B,Kij)),-1);add2(o,solve(matmul(solve(matmul(local.B,Ki)),Kj)));return o;};return {g.k_alpha_alpha,g.k_alpha_range,g.k_range_range,d.alpha_alpha,d.alpha_range,d.range_range,b2(d.alpha_alpha,Ca,g.k_alpha,Ba,g.k_alpha,g.k_alpha_alpha),b2(d.alpha_range,Ca,g.k_range,Br,g.k_alpha,g.k_alpha_range),b2(d.alpha_range,Cr,g.k_alpha,Ba,g.k_range,g.k_alpha_range),b2(d.range_range,Cr,g.k_range,Br,g.k_range,g.k_range_range)};}
BSecondDerivatives compute_b_second_derivatives(GpuContext&c,const WorkerInput&w,const Matrix&knots,const GlobalParameters&p){const auto global=make_p_global_derivative_workspace(c,knots,p);const auto cnk=make_cnk_derivative_workspace(c,w,knots,p);return compute_b_second_derivatives(c,w,knots,p,global,cnk);}
}
