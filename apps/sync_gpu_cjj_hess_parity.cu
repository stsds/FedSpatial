#include "fedspatial/gpu_context.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace fedspatial;

static bool load(const std::string& name, Matrix* value) {
  return read_matrix("sync_gpu/tests/fixtures/sync_gpu/" + name, value);
}
static double max_rel(const Matrix& actual, const Matrix& oracle) {
  double error=0.;
  for (size_t i=0;i<actual.a.size();++i)
    error=std::max(error,std::abs(actual.a[i]-oracle.a[i])/std::max(1.,std::abs(oracle.a[i])));
  return error;
}

int main() {
  Matrix sites(3,2),z(3,1),x(3,1),knots(2,2),gamma(1,1),mu(2,1),sigma(2,2);
  const double site_values[]={0,0,1,.2,.3,1.1}, knot_values[]={0,0,1.5,1.5};
  std::copy(site_values,site_values+6,sites.a.begin()); std::copy(knot_values,knot_values+4,knots.a.begin());
  for(double& value:x.a)value=1.; gamma(0,0)=.15; mu(0,0)=.2; mu(1,0)=-.1;
  sigma(0,0)=.7; sigma(1,1)=.4; sigma(0,1)=sigma(1,0)=.05;
  try {
    GpuContext context(0); WorkerInput worker{sites,z,x}; context.bind_region(worker,knots);
    double aa_abs=0.,ar_abs=0.,ra_abs=0.,rr_abs=0.,aa_rel=0.,ar_rel=0.,ra_rel=0.,rr_rel=0.,symmetry=0.,repeat=0.;
    const auto raw_before=context.raw_h2d_bytes(), d2h_before=context.result_d2h_bytes();
    bool no_active_model_transfer=true;
    for(int point=0;point<7;++point) {
      Matrix parameters,oracle_aa,oracle_ar,oracle_ra,oracle_rr;
      const std::string prefix="DELTA_THETA_POINT"+std::to_string(point);
      if(!load(prefix+"_PARAM.txt",&parameters)||!load(prefix+"_CJJ_HESS_AA.txt",&oracle_aa)||
         !load(prefix+"_CJJ_HESS_AR.txt",&oracle_ar)||!load(prefix+"_CJJ_HESS_RA.txt",&oracle_ra)||!load(prefix+"_CJJ_HESS_RR.txt",&oracle_rr)) return 2;
      const auto raw_before_slice=context.raw_h2d_bytes(), d2h_before_slice=context.result_d2h_bytes();
      const auto slice=compute_device_cjj_second_derivatives(context,3,parameters(1,0),parameters(2,0));
      no_active_model_transfer &= context.raw_h2d_bytes()==raw_before_slice && context.result_d2h_bytes()==d2h_before_slice;
      const auto actual=download_device_cjj_second_derivatives(context,slice);
      const auto repeated=download_device_cjj_second_derivatives(context,compute_device_cjj_second_derivatives(context,3,parameters(1,0),parameters(2,0)));
      aa_abs=std::max(aa_abs,max_abs_diff(actual.aa,oracle_aa)); ar_abs=std::max(ar_abs,max_abs_diff(actual.ar,oracle_ar)); ra_abs=std::max(ra_abs,max_abs_diff(actual.ra,oracle_ra)); rr_abs=std::max(rr_abs,max_abs_diff(actual.rr,oracle_rr));
      aa_rel=std::max(aa_rel,max_rel(actual.aa,oracle_aa)); ar_rel=std::max(ar_rel,max_rel(actual.ar,oracle_ar)); ra_rel=std::max(ra_rel,max_rel(actual.ra,oracle_ra)); rr_rel=std::max(rr_rel,max_rel(actual.rr,oracle_rr));
      symmetry=std::max(symmetry,max_abs_diff(actual.ar,actual.ra));
      repeat=std::max({repeat,max_abs_diff(actual.aa,repeated.aa),max_abs_diff(actual.ar,repeated.ar),max_abs_diff(actual.ra,repeated.ra),max_abs_diff(actual.rr,repeated.rr)});
    }
    const auto d2h_after_oracles=context.result_d2h_bytes();
    const bool pass=std::max({aa_abs,ar_abs,ra_abs,rr_abs})<=1e-10 && std::max({aa_rel,ar_rel,ra_rel,rr_rel})<=1e-10 && symmetry==0. && repeat==0. && context.cjj_second_allocation_count()==1 && no_active_model_transfer && context.raw_h2d_bytes()==raw_before && d2h_before==0 && d2h_after_oracles==static_cast<long long>(7*2*3*sites.rows*sites.rows*sizeof(double));
    std::cout<<"CJJ_AA_MAX_ABS="<<aa_abs<<" CJJ_AR_MAX_ABS="<<ar_abs<<" CJJ_RA_MAX_ABS="<<ra_abs<<" CJJ_RR_MAX_ABS="<<rr_abs<<'\n'
             <<"CJJ_AA_MAX_REL="<<aa_rel<<" CJJ_AR_MAX_REL="<<ar_rel<<" CJJ_RA_MAX_REL="<<ra_rel<<" CJJ_RR_MAX_REL="<<rr_rel<<'\n'
             <<"CJJ_SYMMETRY_MAX_ABS="<<symmetry<<" CJJ_REPEAT_MAX_ABS="<<repeat<<'\n'
             <<"CJJ_ALLOCATIONS="<<context.cjj_second_allocation_count()<<" RAW_H2D_BEFORE="<<raw_before<<" RAW_H2D_AFTER_ACTIVE="<<context.raw_h2d_bytes()<<" ACTIVE_RESULT_D2H="<<d2h_before<<" NO_ACTIVE_MODEL_TRANSFER="<<(no_active_model_transfer?"PASS":"FAIL")<<" RESULT_D2H_AFTER_ORACLES="<<d2h_after_oracles<<'\n'
             <<"CJJ_DEVICE_PARITY="<<(pass?"PASS":"FAIL")<<'\n';
    return pass?0:1;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 2; }
}
