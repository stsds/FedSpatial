#include "fedspatial/gpu_context.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace fedspatial;
static bool load(const std::string& n,Matrix* x){return read_matrix("sync_gpu/tests/fixtures/sync_gpu/"+n,x);}
static double ae(const Matrix&a,const Matrix&b){double x=0;for(size_t i=0;i<a.a.size();++i)x=std::max(x,std::abs(a.a[i]-b.a[i]));return x;}
static double re(const Matrix&a,const Matrix&b){double x=0;for(size_t i=0;i<a.a.size();++i)x=std::max(x,std::abs(a.a[i]-b.a[i])/std::max(1.,std::abs(b.a[i])));return x;}
static double sy(const Matrix&a){double x=0;for(int i=0;i<a.rows;++i)for(int j=0;j<a.cols;++j)x=std::max(x,std::abs(a(i,j)-a(j,i)));return x;}
int main(){
  Matrix s(3,2),z(3,1),x(3,1),k(2,2),g(1,1),mu(2,1),sg(2,2),p;
  double sv[]={0,0,1,.2,.3,1.1},kv[]={0,0,1.5,1.5};std::copy(sv,sv+6,s.a.begin());std::copy(kv,kv+4,k.a.begin());for(double&v:x.a)v=1.;g(0,0)=.15;mu(0,0)=.2;mu(1,0)=-.1;sg(0,0)=.7;sg(1,1)=.4;sg(0,1)=sg(1,0)=.05;
  const char* names[]={"M_GRAD_DELTA","M_GRAD_ALPHA","M_GRAD_RANGE","M_HESS_DD","M_HESS_DA","M_HESS_DR","M_HESS_AD","M_HESS_AA","M_HESS_AR","M_HESS_RD","M_HESS_RA","M_HESS_RR"};
  try { if(!load("DELTA_THETA_POINT0_PARAM.txt",&p))return 2; Matrix want[12];for(int i=0;i<12;++i)if(!load("DELTA_THETA_POINT0_"+std::string(names[i])+".txt",&want[i]))return 2;
    WorkerInput w{s,z,x};GlobalParameters q{g,mu,sg,p(0,0),p(1,0),p(2,0)};GpuContext c(0);c.bind_region(w,k);
    auto h0=c.raw_h2d_bytes(),d0=c.result_d2h_bytes(),b0=c.b_derivative_allocation_count(),a0=c.a_derivative_allocation_count(),cj0=c.cjj_second_allocation_count(),m0=c.m_derivative_allocation_count();
    auto slice=compute_device_m_derivatives(c,q.nu2,q.delta,q.alpha,q.range);auto ah=c.raw_h2d_bytes()-h0,ad=c.result_d2h_bytes()-d0;auto got=download_device_m_derivatives(c,slice);auto again=download_device_m_derivatives(c,compute_device_m_derivatives(c,q.nu2,q.delta,q.alpha,q.range));
    Matrix actual[]={got.first.delta,got.first.alpha,got.first.range,got.second.dd,got.second.da,got.second.dr,got.second.ad,got.second.aa,got.second.ar,got.second.rd,got.second.ra,got.second.rr};
    Matrix repeated[]={again.first.delta,again.first.alpha,again.first.range,again.second.dd,again.second.da,again.second.dr,again.second.ad,again.second.aa,again.second.ar,again.second.rd,again.second.ra,again.second.rr};
    double ma=0,mr=0,sm=0;bool deterministic=true;for(int i=0;i<12;++i){ma=std::max(ma,ae(actual[i],want[i]));mr=std::max(mr,re(actual[i],want[i]));sm=std::max(sm,sy(actual[i]));deterministic=deterministic&&(actual[i].a==repeated[i].a);std::cout<<names[i]<<"_MAX_ABS_ERROR="<<ae(actual[i],want[i])<<"\n";}
    sm=std::max({sm,ae(got.second.da,got.second.ad),ae(got.second.dr,got.second.rd),ae(got.second.ar,got.second.ra)});
    bool pass=ma<=1e-10&&mr<=1e-10&&sm<=1e-10&&deterministic&&ah==0&&ad==0&&c.b_derivative_allocation_count()-b0==1&&c.a_derivative_allocation_count()-a0==1&&c.cjj_second_allocation_count()-cj0==1&&c.m_derivative_allocation_count()-m0==1;
    std::cout<<"M_MAX_REL_ERROR="<<mr<<"\nM_SYMMETRY_MAX_ABS_ERROR="<<sm<<"\nM_EXACT_REPEAT_DETERMINISM="<<(deterministic?"PASS":"FAIL")<<"\nB_ALLOCATION_COUNT_DELTA="<<c.b_derivative_allocation_count()-b0<<"\nA_ALLOCATION_COUNT_DELTA="<<c.a_derivative_allocation_count()-a0<<"\nCJJ_ALLOCATION_COUNT_DELTA="<<c.cjj_second_allocation_count()-cj0<<"\nM_ALLOCATION_COUNT_DELTA="<<c.m_derivative_allocation_count()-m0<<"\nM_ACTIVE_H2D_BYTES="<<ah<<"\nM_ACTIVE_D2H_BYTES="<<ad<<"\nM_DIAGNOSTIC_D2H_BYTES="<<c.result_d2h_bytes()-d0<<"\nM_DEVICE_DERIVATIVE_PARITY="<<(pass?"PASS":"FAIL")<<"\n";return pass?0:1;
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}
}
