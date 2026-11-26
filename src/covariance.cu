#include "fedspatial/gpu_context.hpp"
#include <stdexcept>

namespace fedspatial { namespace {
void check_cov(cudaError_t e, const char* where = "CUDA") { if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e)); }
__global__ void matern_all(const double* x, const double* y, int nx, int ny,
                            int nu2, double alpha, double beta, double* v,
                            double* da, double* db, double* daa, double* dab,
                            double* dbb) {
  int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nx*ny)return;
  int i=t/ny,j=t%ny; double dx=x[2*i]-y[2*j],dy=x[2*i+1]-y[2*j+1];
  double u=sqrt(dx*dx+dy*dy)/beta, e, p, fb, fbb;
  if(nu2==1) { e=exp(-u); p=e; fb=alpha*e*u/beta; fbb=alpha*e*(u*u-2*u)/(beta*beta); }
  else if(nu2==3) { u*=sqrt(3.); e=exp(-u); p=(1.+u)*e; fb=alpha*e*u*u/beta; fbb=alpha*e*(u*u*u-3*u*u)/(beta*beta); }
  else { u*=sqrt(5.); e=exp(-u); p=(1.+u+u*u/3.)*e; fb=alpha*e*(u*u+u*u*u)/(3.*beta); fbb=alpha*e*(u*u*u*u-3.*u*u*u-3.*u*u)/(3.*beta*beta); }
  v[t]=alpha*p; da[t]=p; db[t]=fb; daa[t]=0.; dab[t]=fb/alpha; dbb[t]=fbb;
}
__global__ void cnk_fused(const double*x,const double*y,int nx,int ny,int nu2,double a,double r,double*da,double*dr,double*daa,double*dar,double*drr){int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=nx*ny)return;int i=t/ny,j=t%ny;double dx=x[2*i]-y[2*j],dy=x[2*i+1]-y[2*j+1],u=sqrt(dx*dx+dy*dy)/r,e,p,rb,rbb;if(nu2==1){e=exp(-u);p=e;rb=e*u/r;rbb=a*e*(u*u-2*u)/(r*r);}else if(nu2==3){u*=sqrt(3.);e=exp(-u);p=(1+u)*e;rb=e*u*u/r;rbb=a*e*(u*u*u-3*u*u)/(r*r);}else{u*=sqrt(5.);e=exp(-u);p=(1+u+u*u/3)*e;rb=e*(u*u+u*u*u)/(3*r);rbb=a*e*(u*u*u*u-3*u*u*u-3*u*u)/(3*r*r);}da[t]=p;dr[t]=a*rb;daa[t]=0;dar[t]=rb;drr[t]=rbb;}
CnkDerivativeWorkspace fused_impl(GpuContext&c,const Matrix&x,const Matrix&y,int nu2,double a,double r){size_t n=size_t(x.rows)*y.rows;double *dx,*dy,*da,*dr,*daa,*dar,*drr;cudaMalloc(&dx,x.a.size()*sizeof(double));cudaMalloc(&dy,y.a.size()*sizeof(double));cudaMalloc(&da,n*sizeof(double));cudaMalloc(&dr,n*sizeof(double));cudaMalloc(&daa,n*sizeof(double));cudaMalloc(&dar,n*sizeof(double));cudaMalloc(&drr,n*sizeof(double));cudaMemcpyAsync(dx,x.a.data(),x.a.size()*sizeof(double),cudaMemcpyHostToDevice,c.stream());cudaMemcpyAsync(dy,y.a.data(),y.a.size()*sizeof(double),cudaMemcpyHostToDevice,c.stream());cnk_fused<<<(n+255)/256,256,0,c.stream()>>>(dx,dy,x.rows,y.rows,nu2,a,r,da,dr,daa,dar,drr);CnkDerivativeWorkspace o{Matrix(x.rows,y.rows),Matrix(x.rows,y.rows),Matrix(x.rows,y.rows),Matrix(x.rows,y.rows),Matrix(x.rows,y.rows)};cudaMemcpyAsync(o.alpha.a.data(),da,n*sizeof(double),cudaMemcpyDeviceToHost,c.stream());cudaMemcpyAsync(o.range.a.data(),dr,n*sizeof(double),cudaMemcpyDeviceToHost,c.stream());cudaMemcpyAsync(o.alpha_alpha.a.data(),daa,n*sizeof(double),cudaMemcpyDeviceToHost,c.stream());cudaMemcpyAsync(o.alpha_range.a.data(),dar,n*sizeof(double),cudaMemcpyDeviceToHost,c.stream());cudaMemcpyAsync(o.range_range.a.data(),drr,n*sizeof(double),cudaMemcpyDeviceToHost,c.stream());cudaStreamSynchronize(c.stream());cudaFree(dx);cudaFree(dy);cudaFree(da);cudaFree(dr);cudaFree(daa);cudaFree(dar);cudaFree(drr);return o;}
}
CnkDerivativeWorkspace compute_cnk_derivatives_fused(GpuContext&c,const Matrix&x,const Matrix&y,int nu2,double a,double r){check_cov(cudaSetDevice(c.device_id()));return fused_impl(c,x,y,nu2,a,r);}
MaternDerivatives matern_with_derivatives(GpuContext& context, const Matrix& x,
                                          const Matrix& y, int nu2,
                                          double alpha, double range) {
  PProfileScope profile_scope(context.profile(), "matern_with_derivatives");
  if(x.cols!=2||y.cols!=2||(nu2!=1&&nu2!=3&&nu2!=5)||alpha<=0||range<=0)
    throw std::runtime_error("invalid Matérn Figure 1 arguments");
  check_cov(cudaSetDevice(context.device_id()), "matern cudaSetDevice"); const size_t n=size_t(x.rows)*y.rows;
  double *dx{},*dy{},*v{},*da{},*db{},*daa{},*dab{},*dbb{};
  auto release=[&]{for(double* p:{dx,dy,v,da,db,daa,dab,dbb})if(p)cudaFree(p);};
  try { auto alloc=[&](double**p,size_t count){check_cov(cudaMalloc(p,count*sizeof(double)), "matern cudaMalloc");};
    alloc(&dx,x.a.size());alloc(&dy,y.a.size());for(double**p:{&v,&da,&db,&daa,&dab,&dbb})alloc(p,n);
    check_cov(cudaMemcpyAsync(dx,x.a.data(),x.a.size()*sizeof(double),cudaMemcpyHostToDevice,context.stream()), "matern upload x");
    check_cov(cudaMemcpyAsync(dy,y.a.data(),y.a.size()*sizeof(double),cudaMemcpyHostToDevice,context.stream()), "matern upload y");
    matern_all<<<(n+255)/256,256,0,context.stream()>>>(dx,dy,x.rows,y.rows,nu2,alpha,range,v,da,db,daa,dab,dbb);
    check_cov(cudaGetLastError(), "matern kernel"); MaternDerivatives out{Matrix(x.rows,y.rows),Matrix(x.rows,y.rows),Matrix(x.rows,y.rows),Matrix(x.rows,y.rows),Matrix(x.rows,y.rows),Matrix(x.rows,y.rows),Matrix(x.rows,y.rows)};
    for(auto pair:{std::pair{v,&out.value},std::pair{da,&out.alpha},std::pair{db,&out.range},std::pair{daa,&out.aa},std::pair{dab,&out.ar},std::pair{dab,&out.ra},std::pair{dbb,&out.rr}})
      check_cov(cudaMemcpyAsync(pair.second->a.data(),pair.first,n*sizeof(double),cudaMemcpyDeviceToHost,context.stream()), "matern download");
    check_cov(cudaStreamSynchronize(context.stream()), "matern synchronize");release();return out;
  } catch(...) {release();throw;}
}
}
