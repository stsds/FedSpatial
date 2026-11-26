#include "fedspatial/gpu_context.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace fs=std::filesystem; using namespace fedspatial;
static Matrix load(const fs::path& p){std::ifstream f(p);int r,c;if(!(f>>r>>c))throw std::runtime_error("missing "+p.string());Matrix x(r,c);for(double&v:x.a)f>>v;return x;}
static WorkerInput worker(const fs::path& root,int id){Matrix all=load(root/("worker_"+std::to_string(id)+".txt"));Matrix sites(all.rows,2),z(all.rows,1),x(all.rows,5);for(int i=0;i<all.rows;++i){sites(i,0)=all(i,0);sites(i,1)=all(i,1);z(i,0)=all(i,2);for(int j=0;j<5;++j)x(i,j)=all(i,j+3);}return {sites,z,x};}
static double rel(double a,double b){return std::abs(a-b)/std::max({1.,std::abs(a),std::abs(b)});}
int main(){try{
  const fs::path root="sync_gpu/tests/fixtures/figure1_reference"; const auto knots=load(root/"knots.txt");
  GlobalParameters p{load(root/"p_initial_gamma_new.txt"),load(root/"p_initial_global_mu.txt"),load(root/"p_initial_global_sigma.txt"),.25,1.,.11};
  GpuContext c(0); const auto w=worker(root,0); const auto raw0=c.raw_h2d_bytes(), d2h0=c.result_d2h_bytes(), alloc0=c.delta_theta_reduction_allocation_count();
  const auto first=compute_device_delta_theta_local(c,w,knots,p).summary;
  const auto raw1=c.raw_h2d_bytes(), d2h1=c.result_d2h_bytes(), alloc1=c.delta_theta_reduction_allocation_count();
  const auto second=compute_device_delta_theta_local(c,w,knots,p).summary;
  const bool bytes=std::memcmp(&first,&second,sizeof(first))==0;
  const bool persistent=c.delta_theta_reduction_allocation_count()==alloc1 && alloc1==alloc0+1;
  const bool transfers=c.raw_h2d_bytes()==raw1 && raw1>raw0 && c.result_d2h_bytes()-d2h1==13*long(sizeof(double));
  const auto global=make_p_global_derivative_workspace(c,knots,p);
  const auto oracle=compute_delta_theta_local_legacy_oracle(c,w,knots,p,global);
  double max_abs=std::abs(first.objective-oracle.objective),max_rel=rel(first.objective,oracle.objective),sym=0.;
  std::cout<<std::setprecision(17)<<"OBJECTIVE_MAX_ABS="<<max_abs<<" OBJECTIVE_MAX_REL="<<max_rel<<'\n';
  for(int i=0;i<3;++i){double a=std::abs(first.gradient[i]-oracle.gradient[i]),r=rel(first.gradient[i],oracle.gradient[i]);max_abs=std::max(max_abs,a);max_rel=std::max(max_rel,r);std::cout<<"GRAD_"<<i<<"_MAX_ABS="<<a<<" GRAD_"<<i<<"_MAX_REL="<<r<<'\n';}
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){double a=std::abs(first.hessian[3*i+j]-oracle.hessian[3*i+j]),r=rel(first.hessian[3*i+j],oracle.hessian[3*i+j]);max_abs=std::max(max_abs,a);max_rel=std::max(max_rel,r);sym=std::max(sym,std::abs(first.hessian[3*i+j]-first.hessian[3*j+i]));std::cout<<"HESSIAN_"<<i<<j<<"_MAX_ABS="<<a<<" HESSIAN_"<<i<<j<<"_MAX_REL="<<r<<'\n';}
  const bool pass=max_abs<=1e-9&&max_rel<=1e-9&&sym==0.&&bytes&&persistent&&transfers;
  std::cout<<"DELTA_THETA_EVALUATOR_MAX_ABS="<<max_abs<<"\nDELTA_THETA_EVALUATOR_MAX_REL="<<max_rel<<"\nDELTA_THETA_HESSIAN_SYMMETRY_MAX_ABS="<<sym<<"\nDELTA_THETA_BYTE_IDENTICAL_REPEAT="<<(bytes?"PASS":"FAIL")<<"\nDELTA_THETA_REDUCTION_ALLOCATION_DELTA="<<alloc1-alloc0<<"\nDELTA_THETA_ZERO_REPEAT_ACTIVE_H2D="<<(c.raw_h2d_bytes()==raw1?"PASS":"FAIL")<<"\nDELTA_THETA_COMPACT_SUMMARY_D2H_BYTES="<<d2h1-d2h0<<"\nDELTA_THETA_EVALUATOR_PARITY="<<(pass?"PASS":"FAIL")<<'\n'; return pass?0:1;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
