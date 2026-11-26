#include "fedspatial/figure1_fixture.hpp"
#include "fedspatial/gpu_context.hpp"
#include "fedspatial/nworker_15k_figure1_config.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>

using namespace fedspatial;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
  try {
    const fs::path root=argc==2 ? argv[1] : "results/sync_gpu_2x2k_fixture/scenario_04/rep_000";
    // This executable is a single-worker diagnostic.  UUID masking occurs
    // before the first CUDA runtime call, so logical ordinal zero is AF, not
    // the physical RTX at ordinal zero.
    const std::string mask=NWorker15kFigure1Config::gv100_uuid_0;
    if(setenv("CUDA_VISIBLE_DEVICES",mask.c_str(),1)!=0) throw std::runtime_error("cannot mask GV100s");
    for(const char* gate:{"FEDSPATIAL_MIXED_TRACE_TILED_ACTION","FEDSPATIAL_MIXED_CONTRACTIONS_TILED",
        "FEDSPATIAL_THETA_FIRST_TILED","FEDSPATIAL_THETA_HESSIAN_AA_LOG_TILED",
        "FEDSPATIAL_THETA_HESSIAN_AR_LOG_TILED","FEDSPATIAL_THETA_HESSIAN_RR_LOG_TILED",
        "FEDSPATIAL_THETA_HESSIAN_Q_W_TILED"}) setenv(gate,"1",1);
    setenv("FEDSPATIAL_DELTA_THETA_SUBSTAGE_TIMING","1",1);
    Figure1Fixture fixture; load_figure1_python_fixture(root,&fixture);
    if(fixture.sites.rows!=4000 || fixture.knots.rows!=2000) throw std::runtime_error("requires exact 4k/2k timing fixture");
    const auto workers=partition_figure1_fixture(fixture,2);
    if(workers.size()!=2 || workers[0].sites.rows!=2000) throw std::runtime_error("fixture partition mismatch");
    char bus[32]{}; cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop,0); cudaDeviceGetPCIBusId(bus,sizeof(bus),0);
    if(std::string(prop.name).find("GV100")==std::string::npos) throw std::runtime_error("masked ordinal 0 is not GV100");
    std::cerr<<"DELTA_THETA_PROFILE_GPU logical=0 pci_bus="<<bus<<" name="<<prop.name<<std::endl;
    Matrix gamma(5,1),mu(2000,1),sigma(2000,2000); const double g[]={-1.,2.,3.,-2.,1.};
    for(int i=0;i<5;++i) gamma(i,0)=g[i];
    GlobalParameters p{gamma,mu,sigma,.25,1.,.11,3};
    auto run=[&](const char* mode,bool scalar){
      if(scalar) setenv("FEDSPATIAL_THETA_HESSIAN_SECOND_LOG_TILED","0",1);
      else unsetenv("FEDSPATIAL_THETA_HESSIAN_SECOND_LOG_TILED");
      GpuContext context(0);
      const auto start=std::chrono::steady_clock::now();
      const auto out=compute_device_delta_theta_local(context,workers[0],fixture.knots,p).summary;
      const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      std::cout<<std::setprecision(17)<<"DELTA_THETA_FIXTURE_PROFILE mode="<<mode
               <<" n=2000 m=2000 seconds="<<seconds<<" objective="<<out.objective
               <<" gradient="<<out.gradient[0]<<','<<out.gradient[1]<<','<<out.gradient[2]
               <<" hessian00="<<out.hessian[0]<<std::endl;
      return seconds;
    };
    const double scalar_seconds=run("scalar",true);
    const double tiled_seconds=run("tiled_second_action",false);
    std::cout<<"DELTA_THETA_FIXTURE_PROFILE_SECOND_ACTION_SPEEDUP="
             <<scalar_seconds/tiled_seconds<<std::endl;
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<std::endl; return 2; }
}
