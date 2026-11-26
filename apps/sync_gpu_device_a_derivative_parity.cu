#include "fedspatial/gpu_context.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace fedspatial;

static Matrix right_solve(const Matrix& x,const Matrix& k) { return transpose(solve_spd(k,transpose(x))); }
static void sub(Matrix& x,const Matrix& y) { for(size_t i=0;i<x.a.size();++i) x.a[i]-=y.a[i]; }
static Matrix tri(const Matrix& x,const Matrix& y,const Matrix& z) { return matmul(matmul(x,y),z); }
static Matrix first_b(const Matrix& c,const Matrix& k,const Matrix& b,const Matrix& ci,const Matrix& ki) { Matrix out=right_solve(ci,k); sub(out,right_solve(matmul(b,ki),k)); return out; }
static Matrix a1(const Matrix& ci,const Matrix& b,const Matrix& bi,const Matrix& k,const Matrix& ki) { Matrix out=ci; sub(out,tri(bi,k,transpose(b))); sub(out,tri(b,ki,transpose(b))); sub(out,tri(b,k,transpose(bi))); return out; }
static Matrix a2(const Matrix& cij,const Matrix& b,const Matrix& bi,const Matrix& bj,const Matrix& bij,const Matrix& k,const Matrix& ki,const Matrix& kj,const Matrix& kij) { Matrix out=cij; sub(out,tri(bij,k,transpose(b))); sub(out,tri(bi,kj,transpose(b))); sub(out,tri(bi,k,transpose(bj))); sub(out,tri(bj,ki,transpose(b))); sub(out,tri(b,kij,transpose(b))); sub(out,tri(b,ki,transpose(bj))); sub(out,tri(bj,k,transpose(bi))); sub(out,tri(b,kj,transpose(bi))); sub(out,tri(b,k,transpose(bij))); return out; }
static double abs_error(const Matrix& x,const Matrix& y) { double v=0;for(size_t i=0;i<x.a.size();++i)v=std::max(v,std::abs(x.a[i]-y.a[i]));return v; }
static double rel_error(const Matrix& x,const Matrix& y) { double v=0;for(size_t i=0;i<x.a.size();++i)v=std::max(v,std::abs(x.a[i]-y.a[i])/std::max(1.,std::abs(y.a[i])));return v; }
static double symmetry(const Matrix& x) { double v=0;for(int i=0;i<x.rows;++i)for(int j=0;j<x.cols;++j)v=std::max(v,std::abs(x(i,j)-x(j,i)));return v; }

int main() {
  Matrix sites(3,2), z(3,1), x(3,1), knots(2,2), gamma(1,1), mu(2,1), sigma(2,2);
  const double sv[]={0,0,1,.2,.3,1.1}, kv[]={0,0,1.5,1.5};
  std::copy(sv,sv+6,sites.a.begin()); std::copy(kv,kv+4,knots.a.begin());
  for(double& v:x.a)v=1.; gamma(0,0)=.15; mu(0,0)=.2; mu(1,0)=-.1; sigma(0,0)=.7; sigma(1,1)=.4; sigma(0,1)=sigma(1,0)=.05;
  WorkerInput worker{sites,z,x}; GlobalParameters p{gamma,mu,sigma,1.7,1.2,.8};
  try {
    // The host equations are deliberately confined to this oracle context.
    GpuContext oracle_context(0); const auto local=compute_mu_sigma_local(oracle_context,worker,knots,p);
    const auto d= matern_with_derivatives(oracle_context,sites,knots,p.nu2,p.alpha,p.range);
    const auto k= matern_with_derivatives(oracle_context,knots,knots,p.nu2,p.alpha,p.range);
    const auto j= matern_with_derivatives(oracle_context,sites,sites,p.nu2,p.alpha,p.range);
    const auto h=compute_b_second_derivatives(oracle_context,worker,knots,p);
    const Matrix ba=first_b(d.value,k.value,local.B,d.alpha,k.alpha), br=first_b(d.value,k.value,local.B,d.range,k.range);
    AFirstDerivatives want_first{a1(j.alpha,local.B,ba,k.value,k.alpha),a1(j.range,local.B,br,k.value,k.range)};
    ASecondDerivatives want_second{a2(j.aa,local.B,ba,ba,h.baa,k.value,k.alpha,k.alpha,h.kaa),a2(j.ar,local.B,ba,br,h.bar,k.value,k.alpha,k.range,h.kar),a2(j.ra,local.B,br,ba,h.bra,k.value,k.range,k.alpha,h.kar),a2(j.rr,local.B,br,br,h.brr,k.value,k.range,k.range,h.krr)};

    GpuContext c(0); c.bind_region(worker,knots);
    const auto h2d0=c.raw_h2d_bytes(), d2h0=c.result_d2h_bytes(), b0=c.b_derivative_allocation_count(), a0=c.a_derivative_allocation_count(), cjj0=c.cjj_second_allocation_count();
    const auto slice=compute_device_a_derivatives(c,p.nu2,p.alpha,p.range);
    const auto h2d_active=c.raw_h2d_bytes()-h2d0, d2h_active=c.result_d2h_bytes()-d2h0;
    const auto first=download_device_a_derivatives(c,slice);
    const auto repeat=download_device_a_derivatives(c,compute_device_a_derivatives(c,p.nu2,p.alpha,p.range));
    const bool deterministic=first.first.alpha.a==repeat.first.alpha.a && first.first.range.a==repeat.first.range.a && first.second.aa.a==repeat.second.aa.a && first.second.ar.a==repeat.second.ar.a && first.second.ra.a==repeat.second.ra.a && first.second.rr.a==repeat.second.rr.a;
    const double aa=abs_error(first.second.aa,want_second.aa), ar=abs_error(first.second.ar,want_second.ar), ra=abs_error(first.second.ra,want_second.ra), rr=abs_error(first.second.rr,want_second.rr), fa=abs_error(first.first.alpha,want_first.alpha), fr=abs_error(first.first.range,want_first.range);
    const double rel=std::max({rel_error(first.first.alpha,want_first.alpha),rel_error(first.first.range,want_first.range),rel_error(first.second.aa,want_second.aa),rel_error(first.second.ar,want_second.ar),rel_error(first.second.ra,want_second.ra),rel_error(first.second.rr,want_second.rr)});
    const double sym=std::max({symmetry(first.first.alpha),symmetry(first.first.range),symmetry(first.second.aa),symmetry(first.second.ar),symmetry(first.second.ra),symmetry(first.second.rr),abs_error(first.second.ar,first.second.ra)});
    const bool pass=std::max({fa,fr,aa,ar,ra,rr})<=1e-10 && rel<=1e-10 && sym<=1e-10 && deterministic && h2d_active==0 && d2h_active==0 && c.b_derivative_allocation_count()-b0==1 && c.a_derivative_allocation_count()-a0==1 && c.cjj_second_allocation_count()-cjj0==1;
    std::cout<<"A_FIRST_ALPHA_MAX_ABS_ERROR="<<fa<<"\nA_FIRST_RANGE_MAX_ABS_ERROR="<<fr<<"\nA_SECOND_AA_MAX_ABS_ERROR="<<aa<<"\nA_SECOND_AR_MAX_ABS_ERROR="<<ar<<"\nA_SECOND_RA_MAX_ABS_ERROR="<<ra<<"\nA_SECOND_RR_MAX_ABS_ERROR="<<rr<<"\nA_MAX_REL_ERROR="<<rel<<"\nA_SYMMETRY_MAX_ABS_ERROR="<<sym<<"\nA_EXACT_REPEAT_DETERMINISM="<<(deterministic?"PASS":"FAIL")<<"\nB_ALLOCATION_COUNT_DELTA="<<c.b_derivative_allocation_count()-b0<<"\nA_ALLOCATION_COUNT_DELTA="<<c.a_derivative_allocation_count()-a0<<"\nCJJ_ALLOCATION_COUNT_DELTA="<<c.cjj_second_allocation_count()-cjj0<<"\nA_ACTIVE_H2D_BYTES="<<h2d_active<<"\nA_ACTIVE_D2H_BYTES="<<d2h_active<<"\nA_DIAGNOSTIC_D2H_BYTES="<<c.result_d2h_bytes()-d2h0<<"\nA_DEVICE_DERIVATIVE_PARITY="<<(pass?"PASS":"FAIL")<<"\n";
    return pass?0:1;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; }
}
