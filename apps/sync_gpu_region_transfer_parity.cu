#include "fedspatial/gpu_context.hpp"
#include <chrono>
#include <iostream>

using namespace fedspatial;
int main() {
  Matrix sites(3,2), z(3,1), x(3,1), knots(2,2), gamma(1,1), mu(2,1), sigma(2,2);
  const double sv[]={0.,0.,1.,.2,.3,1.1}, zv[]={1.2,-.4,.7}, kv[]={0.,0.,1.5,1.5};
  for(int i=0;i<6;++i) sites.a[i]=sv[i]; for(int i=0;i<3;++i){z.a[i]=zv[i];x.a[i]=1.;}
  for(int i=0;i<4;++i) knots.a[i]=kv[i]; gamma(0,0)=.15; mu(0,0)=.2; mu(1,0)=-.1; sigma(0,0)=.7; sigma(1,1)=.4; sigma(0,1)=sigma(1,0)=.05;
  WorkerInput worker{sites,z,x}; GlobalParameters p{gamma,mu,sigma,1.7,1.2,.8};
  try { GpuContext context(0); context.bind_region(worker,knots); const auto initial=context.raw_h2d_bytes();
    const auto start=std::chrono::steady_clock::now();
    const auto first=compute_mu_sigma_local(context,worker,knots,p);
    const auto end=std::chrono::steady_clock::now(); const auto model_after_first=context.result_d2h_bytes();
    const auto second=compute_mu_sigma_local(context,worker,knots,p);
    // This host-facing compatibility call returns gamma (q=1) and XX (q*q=1):
    // two FP64 scalars.  It is intentionally separate from the cached
    // mu/sigma compatibility extraction above.
    (void)compute_gamma_local(context,worker,knots,p);
    const auto after=context.raw_h2d_bytes(), model_after_repeat=context.result_d2h_bytes();
    constexpr long long kMuSigmaCompatibilityD2H =
        (3 * 3 * 2 + 3 * 3 + 2 * 2 + 2 * 3 + 2 + 4) * sizeof(double);
    constexpr long long kGammaCompactD2H = (1 + 1 * 1) * sizeof(double);
    const bool pass=initial==after &&
        model_after_first==kMuSigmaCompatibilityD2H &&
        model_after_repeat==kMuSigmaCompatibilityD2H+kGammaCompactD2H &&
        max_abs_diff(first.M,second.M)==0.;
    std::cout<<"RAW_H2D_INITIAL="<<initial<<" RAW_H2D_AFTER="<<after
             <<" MODEL_D2H_FIRST="<<model_after_first<<" MODEL_D2H_REPEAT="<<model_after_repeat
             <<" MU_SIGMA_COMPATIBILITY_D2H="<<kMuSigmaCompatibilityD2H
             <<" GAMMA_COMPACT_D2H="<<kGammaCompactD2H
             <<" ACTIVE_LOCAL_FIRST_MS="<<std::chrono::duration<double,std::milli>(end-start).count()
             <<"\nREGION_TRANSFER_PARITY="<<(pass?"PASS":"FAIL")<<'\n'; return pass?0:1;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 2; }
}
