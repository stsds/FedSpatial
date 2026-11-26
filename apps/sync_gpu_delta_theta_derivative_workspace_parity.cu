#include "fedspatial/gpu_context.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace fedspatial;
namespace {
// Unit floor makes the relative metric meaningful for analytic zeros while
// max-abs separately reports absolute parity at those entries.
double rel(const Matrix& a,const Matrix& b) { double x=0.; for(size_t i=0;i<a.a.size();++i) x=std::max(x,std::abs(a.a[i]-b.a[i])/std::max(1.,std::abs(b.a[i]))); return x; }
double sym(const Matrix& a) { double x=0.; for(int i=0;i<a.rows;++i)for(int j=0;j<a.cols;++j)x=std::max(x,std::abs(a(i,j)-a(j,i)));return x; }
bool exact(const Matrix&a,const Matrix&b){return a.rows==b.rows&&a.cols==b.cols&&a.a==b.a;}
Matrix right_solve(const Matrix&a,const Matrix&k){return transpose(solve_spd(k,transpose(a)));}
void add(Matrix&a,const Matrix&b,double s=1.){for(size_t i=0;i<a.a.size();++i)a.a[i]+=s*b.a[i];}
struct Oracle { std::array<Matrix,3> bi,ri,qi; std::array<std::array<Matrix,3>,3> bij,rij,qij; };
void deriv(const Matrix& x,const Matrix& y,double alpha,double range,Matrix& da,Matrix& dr,Matrix& daa,Matrix& dar,Matrix& drr) {
  for(int i=0;i<x.rows;++i)for(int j=0;j<y.rows;++j){ double dx=x(i,0)-y(j,0),dy=x(i,1)-y(j,1),u=std::sqrt(3.)*std::sqrt(dx*dx+dy*dy)/range,e=std::exp(-u); da(i,j)=(1+u)*e;dr(i,j)=alpha*u*u*e/range;daa(i,j)=0.;dar(i,j)=u*u*e/range;drr(i,j)=alpha*e*(u*u*u-3*u*u)/(range*range); }
}
Oracle oracle(const Matrix& sites,const Matrix& knots,const Matrix& z,const Matrix& x,const Matrix& gamma,const Matrix& mu,const Matrix& sigma,double delta,double alpha,double range) {
  (void)delta; const int n=sites.rows,m=knots.rows; auto local=make_local_p(sites,knots,z,x,gamma,1.7,alpha,range); Matrix ca(n,m),cr(n,m),caa(n,m),car(n,m),crr(n,m),ka(m,m),kr(m,m),kaa(m,m),kar(m,m),krr(m,m); deriv(sites,knots,alpha,range,ca,cr,caa,car,crr);deriv(knots,knots,alpha,range,ka,kr,kaa,kar,krr);
  Oracle o; for(int i=0;i<3;++i){o.bi[i]=Matrix(n,m);o.ri[i]=Matrix(n,1);o.qi[i]=Matrix(n,n);for(int j=0;j<3;++j){o.bij[i][j]=Matrix(n,m);o.rij[i][j]=Matrix(n,1);o.qij[i][j]=Matrix(n,n);}}
  o.bi[1]=right_solve(ca,local.K);add(o.bi[1],right_solve(matmul(local.B,ka),local.K),-1.);o.bi[2]=right_solve(cr,local.K);add(o.bi[2],right_solve(matmul(local.B,kr),local.K),-1.);
  Matrix* c1[3]={nullptr,&ca,&cr}; Matrix* k1[3]={nullptr,&ka,&kr}; Matrix* c2[3][3]={{nullptr,nullptr,nullptr},{nullptr,&caa,&car},{nullptr,&car,&crr}}; Matrix* k2[3][3]={{nullptr,nullptr,nullptr},{nullptr,&kaa,&kar},{nullptr,&kar,&krr}};
  for(int i=1;i<3;++i)for(int j=1;j<3;++j){auto v=right_solve(*c2[i][j],local.K);add(v,right_solve(matmul(right_solve(*c1[i],local.K),*k1[j]),local.K),-1.);add(v,right_solve(matmul(o.bi[j],*k1[i]),local.K),-1.);add(v,right_solve(matmul(local.B,*k2[i][j]),local.K),-1.);add(v,right_solve(matmul(right_solve(matmul(local.B,*k1[i]),local.K),*k1[j]),local.K));o.bij[i][j]=v;}
  for(int i=0;i<3;++i){o.ri[i]=matmul(o.bi[i],mu);o.qi[i]=matmul(matmul(o.bi[i],sigma),transpose(local.B));add(o.qi[i],matmul(matmul(local.B,sigma),transpose(o.bi[i])));for(int j=0;j<3;++j){o.rij[i][j]=matmul(o.bij[i][j],mu);o.qij[i][j]=matmul(matmul(o.bij[i][j],sigma),transpose(local.B));add(o.qij[i][j],matmul(matmul(local.B,sigma),transpose(o.bij[i][j])));add(o.qij[i][j],matmul(matmul(o.bi[i],sigma),transpose(o.bi[j])));add(o.qij[i][j],matmul(matmul(o.bi[j],sigma),transpose(o.bi[i])));}}
  return o;
}
void report(const char* name,const Matrix&a,const Matrix&b,double& aa,double& rr){const double x=max_abs_diff(a,b),y=rel(a,b);aa=std::max(aa,x);rr=std::max(rr,y);std::cout<<name<<"_MAX_ABS="<<x<<'\n'<<name<<"_MAX_REL="<<y<<'\n';}
}
int main(){try{
  constexpr int n=3,m=2; Matrix sites(n,2),z(n,1),x(n,1),knots(m,2),gamma(1,1),mu(m,1),sigma(m,m); sites.a={0,.1,.8,.3,.2,.9};knots.a={0,0,1,1};z.a={.3,-.2,.7};x.a={1,1,1};gamma(0,0)=.15;mu(0,0)=.2;mu(1,0)=-.1;sigma(0,0)=.7;sigma(1,1)=.4;sigma(0,1)=sigma(1,0)=.05; WorkerInput worker{sites,z,x};GlobalParameters p{gamma,mu,sigma,1.7,1.2,.8,3};GpuContext c(0);auto state=build_device_mu_sigma_local(c,worker,knots,p);auto want=oracle(sites,knots,z,x,gamma,mu,sigma,p.delta,p.alpha,p.range);
  const auto h2d=c.raw_h2d_bytes(),d2h=c.result_d2h_bytes(),alloc=c.delta_theta_derivative_workspace_allocation_count(),balloc=c.b_derivative_allocation_count();auto w=build_device_delta_theta_derivative_workspace(c,state);const bool no_active_xfer=c.raw_h2d_bytes()==h2d&&c.result_d2h_bytes()==d2h;auto got=download_device_delta_theta_derivative_workspace_for_test(c,w);const auto after_download=c.result_d2h_bytes();double aa=0.,rr=0.,matrix_sym=0.,parameter_sym=0.;
  for(int i=0;i<3;++i){report((std::string("B_I_")+std::to_string(i)).c_str(),got.B_i[i],want.bi[i],aa,rr);report((std::string("R_I_")+std::to_string(i)).c_str(),got.r_i[i],want.ri[i],aa,rr);report((std::string("Q_I_")+std::to_string(i)).c_str(),got.Q_i[i],want.qi[i],aa,rr);matrix_sym=std::max(matrix_sym,sym(got.Q_i[i]));for(int j=0;j<3;++j){report((std::string("B_IJ_")+std::to_string(i)+std::to_string(j)).c_str(),got.B_ij[i][j],want.bij[i][j],aa,rr);report((std::string("R_IJ_")+std::to_string(i)+std::to_string(j)).c_str(),got.r_ij[i][j],want.rij[i][j],aa,rr);report((std::string("Q_IJ_")+std::to_string(i)+std::to_string(j)).c_str(),got.Q_ij[i][j],want.qij[i][j],aa,rr);matrix_sym=std::max(matrix_sym,sym(got.Q_ij[i][j]));parameter_sym=std::max({parameter_sym,max_abs_diff(got.B_ij[i][j],got.B_ij[j][i]),max_abs_diff(got.r_ij[i][j],got.r_ij[j][i]),max_abs_diff(got.Q_ij[i][j],got.Q_ij[j][i])});}}
  auto repeat=build_device_delta_theta_derivative_workspace(c,state);const bool alloc_ok=c.delta_theta_derivative_workspace_allocation_count()==alloc+1&&c.b_derivative_allocation_count()==balloc+1;const bool repeat_xfer=c.raw_h2d_bytes()==h2d&&c.result_d2h_bytes()==after_download;auto again=download_device_delta_theta_derivative_workspace_for_test(c,repeat);bool deterministic=true;for(int i=0;i<3;++i)for(int j=0;j<3;++j)deterministic=deterministic&&exact(got.B_i[i],again.B_i[i])&&exact(got.r_i[i],again.r_i[i])&&exact(got.Q_i[i],again.Q_i[i])&&exact(got.B_ij[i][j],again.B_ij[i][j])&&exact(got.r_ij[i][j],again.r_ij[i][j])&&exact(got.Q_ij[i][j],again.Q_ij[i][j]);const bool ok=aa<=1e-10&&rr<=1e-9&&matrix_sym<=1e-12&&parameter_sym<=1e-10&&deterministic&&alloc_ok&&no_active_xfer&&repeat_xfer;std::cout<<"DELTA_THETA_DERIVATIVE_MAX_ABS="<<aa<<"\nDELTA_THETA_DERIVATIVE_MAX_REL="<<rr<<"\nDELTA_THETA_DERIVATIVE_MATRIX_SYMMETRY="<<matrix_sym<<"\nDELTA_THETA_DERIVATIVE_PARAMETER_SYMMETRY="<<parameter_sym<<"\nDELTA_THETA_DERIVATIVE_EXACT_REPEAT="<<(deterministic?"PASS":"FAIL")<<"\nDELTA_THETA_DERIVATIVE_PERSISTENT_ALLOCATIONS="<<(alloc_ok?"PASS":"FAIL")<<"\nDELTA_THETA_DERIVATIVE_ZERO_ACTIVE_H2D_D2H="<<(no_active_xfer&&repeat_xfer?"PASS":"FAIL")<<"\nDELTA_THETA_DERIVATIVE_WORKSPACE_PARITY="<<(ok?"PASS":"FAIL")<<'\n';return ok?0:1;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
