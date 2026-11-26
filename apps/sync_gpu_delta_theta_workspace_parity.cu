#include "fedspatial/gpu_context.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace fedspatial;
namespace {
double max_rel(const Matrix& got, const Matrix& want) {
  double out=0.;
  for (size_t i=0;i<got.a.size();++i)
    out=std::max(out,std::abs(got.a[i]-want.a[i])/std::max(1e-300,std::abs(want.a[i])));
  return out;
}
double symmetry_error(const Matrix& x) {
  double out=0.;
  for(int i=0;i<x.rows;++i) for(int j=0;j<x.cols;++j)
    out=std::max(out,std::abs(x(i,j)-x(j,i)));
  return out;
}
bool exactly_equal(const Matrix& a, const Matrix& b) { return a.rows==b.rows && a.cols==b.cols && a.a==b.a; }
void report(const char* name, const Matrix& got, const Matrix& want, double& abs, double& rel) {
  const double a=max_abs_diff(got,want),r=max_rel(got,want);
  abs=std::max(abs,a); rel=std::max(rel,r);
  std::cout<<name<<"_MAX_ABS="<<a<<'\n'<<name<<"_MAX_REL="<<r<<'\n';
}
}

int main() { try {
  constexpr int n=3,m=2;
  Matrix sites(n,2),z(n,1),x(n,1),knots(m,2),gamma(1,1),mu(m,1),sigma(m,m);
  const double site_values[]={0,.1,.8,.3,.2,.9}, knot_values[]={0,0,1,1};
  std::copy(site_values,site_values+6,sites.a.begin()); std::copy(knot_values,knot_values+4,knots.a.begin());
  z.a={.3,-.2,.7}; for(double& v:x.a)v=1.; gamma(0,0)=.15;
  mu(0,0)=.2; mu(1,0)=-.1; sigma(0,0)=.7; sigma(1,1)=.4; sigma(0,1)=sigma(1,0)=.05;
  WorkerInput worker{sites,z,x}; GlobalParameters p{gamma,mu,sigma,1.7,1.2,.8,3};
  GpuContext context(0);
  const auto local=build_device_mu_sigma_local(context,worker,knots,p);
  const long long h2d_before=context.raw_h2d_bytes(), d2h_before=context.result_d2h_bytes();
  const auto allocations_before=context.delta_theta_workspace_allocation_count();
  const auto workspace=build_device_delta_theta_workspace(context,local);
  const bool active_transfers=context.raw_h2d_bytes()==h2d_before && context.result_d2h_bytes()==d2h_before;
  const auto first=download_device_delta_theta_workspace_for_test(context,workspace);
  const long long d2h_after_first_download=context.result_d2h_bytes();

  // CPU oracle mirrors the legacy delta_theta definitions and their order.
  const auto cpu=make_local_p(sites,knots,z,x,gamma,p.delta,p.alpha,p.range);
  const Matrix e=subtract(z,matmul(x,gamma));
  Matrix want_r=matmul(cpu.B,mu); for(int i=0;i<n;++i) want_r(i,0)-=e(i,0);
  const Matrix want_u=solve_spd(cpu.M,want_r);
  const Matrix want_W=solve_spd(cpu.M,identity(n));
  const Matrix want_Q=matmul(matmul(cpu.B,sigma),transpose(cpu.B));
  double abs=0.,rel=0.; report("DELTA_THETA_R",first.r,want_r,abs,rel);
  report("DELTA_THETA_U",first.u,want_u,abs,rel); report("DELTA_THETA_W",first.W,want_W,abs,rel);
  report("DELTA_THETA_Q",first.Q,want_Q,abs,rel);
  const double q_symmetry=symmetry_error(first.Q);

  const auto repeated=build_device_delta_theta_workspace(context,local);
  const bool allocation_stable=context.delta_theta_workspace_allocation_count()==allocations_before+1;
  const bool repeat_transfers=context.raw_h2d_bytes()==h2d_before &&
                              context.result_d2h_bytes()==d2h_after_first_download;
  const auto second=download_device_delta_theta_workspace_for_test(context,repeated);
  const bool deterministic=exactly_equal(first.r,second.r) && exactly_equal(first.u,second.u) &&
                           exactly_equal(first.W,second.W) && exactly_equal(first.Q,second.Q);
  const bool ok=abs<=1e-10 && rel<=1e-9 && q_symmetry<=1e-12 && deterministic &&
                allocation_stable && active_transfers && repeat_transfers;
  std::cout<<"DELTA_THETA_WORKSPACE_MAX_ABS="<<abs<<'\n'
           <<"DELTA_THETA_WORKSPACE_MAX_REL="<<rel<<'\n'
           <<"DELTA_THETA_Q_SYMMETRY_MAX_ABS="<<q_symmetry<<'\n'
           <<"DELTA_THETA_WORKSPACE_EXACT_REPEAT="<<(deterministic?"PASS":"FAIL")<<'\n'
           <<"DELTA_THETA_WORKSPACE_PERSISTENT_ALLOCATIONS="<<(allocation_stable?"PASS":"FAIL")<<'\n'
           <<"DELTA_THETA_WORKSPACE_ZERO_ACTIVE_H2D_D2H="<<(active_transfers?"PASS":"FAIL")<<'\n'
           <<"DELTA_THETA_WORKSPACE_PARITY="<<(ok?"PASS":"FAIL")<<'\n';
  return ok?0:1;
} catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 2; } }
