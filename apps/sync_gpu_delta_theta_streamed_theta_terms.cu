#include "fedspatial/gpu_context.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace fedspatial;
static double dot(const Matrix&a,const Matrix&b){double x=0.;for(size_t i=0;i<a.a.size();++i)x+=a.a[i]*b.a[i];return x;}
static double tr(const Matrix&a){double x=0.;for(int i=0;i<a.rows;++i)x+=a(i,i);return x;}
static double logdet(const Matrix&a){const Matrix l=cholesky_lower(a);double x=0.;for(int i=0;i<a.rows;++i)x+=2.*std::log(l(i,i));return x;}
static double objective(const WorkerInput&w,const Matrix&k,const GlobalParameters&p){const auto q=make_local_p(w.sites,k,w.z,w.x,p.gamma,p.delta,p.alpha,p.range);const Matrix e=subtract(matmul(w.x,p.gamma),w.z),r=subtract(matmul(q.B,p.mu),e),u=solve_spd(q.M,r),Q=matmul(matmul(q.B,p.sigma),transpose(q.B));return -w.sites.rows*std::log(p.delta)+logdet(q.M)+p.delta*(dot(r,u)+tr(matmul(inverse_spd(q.M),Q)));}
static void row(const char*name,const std::array<double,6>&cpu,const std::array<double,6>&gpu){static const char*labels[]={"tr(W M_i)","2 r_i^T u","u^T M_i u","tr(W Q_i)","tr(W M_i W Q)","assembled gradient"};for(int j=0;j<6;++j)std::cout<<name<<' '<<labels[j]<<" CPU_DENSE="<<cpu[j]<<" GPU="<<gpu[j]<<" ABS="<<std::abs(cpu[j]-gpu[j])<<'\n';}
int main(){try{
  Matrix k(2,2),s(3,2),z(3,1),x(3,1),g(1,1),mu(2,1),sig(2,2),point,want;
  double av[]={0,0,1,.2,.3,1.1},kv[]={0,0,1.5,1.5},zv[]={1.2,-.4,.7};std::copy(av,av+6,s.a.begin());std::copy(kv,kv+4,k.a.begin());std::copy(zv,zv+3,z.a.begin());for(double&v:x.a)v=1.;g(0,0)=.15;mu(0,0)=.2;mu(1,0)=-.1;sig(0,0)=.7;sig(1,1)=.4;sig(0,1)=sig(1,0)=.05;
  if(!read_matrix("sync_gpu/tests/fixtures/sync_gpu/DELTA_THETA_POINT0_PARAM.txt",&point)||!read_matrix("sync_gpu/tests/fixtures/sync_gpu/DELTA_THETA_POINT0_GRAD.txt",&want))throw std::runtime_error("missing autograd fixture");
  GlobalParameters p{g,mu,sig,point(0,0),point(1,0),point(2,0)};WorkerInput w{s,z,x};GpuContext c(0);const auto state=build_device_mu_sigma_local(c,w,k,p);const auto fixed=download_device_delta_theta_workspace_for_test(c,build_device_delta_theta_workspace(c,state));const auto d=download_device_delta_theta_derivative_workspace_for_test(c,build_device_delta_theta_derivative_workspace(c,state));const auto md=download_device_m_derivatives(c,compute_device_m_derivatives(c,p.nu2,p.delta,p.alpha,p.range)).first;const auto gpu=compute_device_delta_theta_theta_first_diagnostics_for_test(c,w,k,p);
  const auto local=make_local_p(s,k,z,x,g,p.delta,p.alpha,p.range);Matrix v=matmul(fixed.W,local.B);for(double&x:v.a)x*=p.delta;const Matrix Mi[]={md.delta,md.alpha,md.range};
  std::cout<<std::setprecision(17)<<"THETA_SCALAR_AUDIT_N=3 V=delta*W*B; B_i/A_i/r_i/Q_i are dense downloads\n";
  for(int parameter=1;parameter<3;++parameter){Matrix Ai=Mi[parameter];for(double&value:Ai.a)value/=p.delta;std::array<double,6> cpu={tr(matmul(fixed.W,Mi[parameter])),2.*dot(d.r_i[parameter],fixed.u),dot(fixed.u,matmul(Mi[parameter],fixed.u)),tr(matmul(fixed.W,d.Q_i[parameter])),tr(matmul(matmul(matmul(fixed.W,Mi[parameter]),fixed.W),fixed.Q)),0.};cpu[5]=cpu[0]+p.delta*(cpu[1]-cpu[2]+cpu[3]-cpu[4]);row(parameter==1?"ALPHA":"RANGE",cpu,gpu.values[parameter-1]);
    const double compact_t1=2.*tr(matmul(matmul(transpose(d.B_i[parameter]),v),p.sigma))/p.delta;const double compact_t2=tr(matmul(matmul(matmul(transpose(v),Ai),v),p.sigma))/p.delta;std::cout<<(parameter==1?"ALPHA":"RANGE")<<" IDENTITY tr(WQi)_dense="<<cpu[3]<<" compact_V="<<compact_t1<<" ABS="<<std::abs(cpu[3]-compact_t1)<<"; tr(WMiWQ)_dense="<<cpu[4]<<" compact_V="<<compact_t2<<" ABS="<<std::abs(cpu[4]-compact_t2)<<'\n';
    const double h=1.e-6;GlobalParameters plus=p,minus=p;if(parameter==1){plus.alpha+=h;minus.alpha-=h;}else{plus.range+=h;minus.range-=h;}const double fd=(objective(w,k,plus)-objective(w,k,minus))/(2*h);std::cout<<(parameter==1?"ALPHA":"RANGE")<<" LOCAL_FD="<<fd<<" AUTOGRAD="<<want(parameter,0)<<" FINAL_GPU="<<gpu.values[parameter-1][5]<<'\n';}
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
