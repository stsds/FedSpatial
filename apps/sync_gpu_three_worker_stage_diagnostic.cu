#include "fedspatial/gpu_context.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// Bounded timing diagnostic: exactly three FP64 workers on devices 0/1/2.
// It deliberately uses n=128 and m=64, so it cannot become a Figure-1 pilot.
using namespace fedspatial;
namespace {
constexpr int kWorkers=3, kN=128, kM=64, kQ=5;
WorkerInput worker_input(int id) {
  WorkerInput w{Matrix(kN,2),Matrix(kN,1),Matrix(kN,kQ)};
  for (int r=0;r<kN;++r) {
    w.sites(r,0)=std::fmod(.071*(r+17*id),1.);
    w.sites(r,1)=std::fmod(.113*(r+11*id),1.);
    w.z(r,0)=std::sin(.03*(r+id));
    for(int c=0;c<kQ;++c) w.x(r,c)=(c==0?1.:std::cos(.01*(c+1)*(r+id)));
  }
  return w;
}
Matrix knot_input() { Matrix k(kM,2); for(int r=0;r<kM;++r) { k(r,0)=std::fmod(.137*r,1.); k(r,1)=std::fmod(.193*r,1.); } return k; }
GlobalParameters initial() { Matrix g(kQ,1),mu(kM,1),sigma(kM,kM); const double v[kQ]={-1,2,3,-2,1}; for(int i=0;i<kQ;++i)g(i,0)=v[i]; for(int i=0;i<kM;++i)sigma(i,i)=1.; return {g,mu,sigma,.25,1.,.11,3}; }
}
int main() { try {
  int devices{}; if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices<3) throw std::runtime_error("requires devices 0,1,2");
  std::vector<WorkerInput> workers; for(int i=0;i<kWorkers;++i)workers.push_back(worker_input(i));
  NWorkerPSynchronousExecutor executor({0,1,2});
  const auto out=executor.run_round_with_checkpoints(workers,knot_input(),initial());
  const auto&t=out.timing;
  std::cout<<std::setprecision(9)<<"BOUNDED_STAGE_CONFIGURATION workers=3 devices=0,1,2 n=128 m=64 q=5 FP64 rounds=1\n"
           <<"BOUNDED_STAGE_SECONDS mu_sigma="<<t.mu_sigma_local_seconds<<" gamma="<<t.gamma_local_seconds
           <<" delta_theta="<<t.delta_theta_local_seconds<<" aggregate="<<t.aggregation_seconds
           <<" global="<<t.global_update_seconds<<" total="<<t.total_seconds<<"\n"
           <<"BOUNDED_STAGE_TRANSFERS raw_h2d="<<t.raw_h2d_bytes<<" model_h2d="<<t.model_h2d_bytes
           <<" model_d2h="<<t.model_d2h_bytes<<" peer="<<t.peer_summary_bytes<<" pinned="<<t.pinned_summary_bytes<<"\n"
           <<"BOUNDED_STAGE_STATUS="<<(t.raw_data_stationary?"PASS":"FAIL")<<std::endl;
  return t.raw_data_stationary?0:1;
 } catch(const std::exception&e) { std::cerr<<"BOUNDED_STAGE_STATUS=FAIL reason="<<e.what()<<std::endl; return 2; } }
