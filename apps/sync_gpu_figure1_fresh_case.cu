#include "fedspatial/figure1_fixture.hpp"
#include "fedspatial/gpu_context.hpp"
#include "fedspatial/nworker_15k_figure1_config.hpp"

#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace fedspatial;
namespace fs = std::filesystem;

static std::vector<int> parse_devices(const std::string& text) {
  std::vector<int> out; std::stringstream stream(text); std::string item;
  while (std::getline(stream, item, ',')) out.push_back(std::stoi(item));
  if (out.empty()) throw std::runtime_error("empty worker device list"); return out;
}
static std::string lowercase(std::string value) { for(char& c:value)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return value; }
static void require_target_gv100_mapping(std::vector<int>* devices) {
  using Target=NWorker15kFigure1Config;
  // CUDA_VISIBLE_DEVICES accepts UUIDs.  Set it before the first CUDA call so
  // logical ordinals 0/1 are the two physical GV100s, never the RTX.
  const std::string visible=std::string(Target::gv100_uuid_0)+","+Target::gv100_uuid_1;
  if (setenv("CUDA_VISIBLE_DEVICES",visible.c_str(),1) != 0) throw std::runtime_error("cannot set CUDA_VISIBLE_DEVICES for GV100 target");
  int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess || count != 2) throw std::runtime_error("GV100 target requires exactly two visible CUDA devices");
  const char* wanted_bus[2]={Target::gv100_bus_0,Target::gv100_bus_1};
  for(int ordinal=0;ordinal<2;++ordinal) {
    cudaDeviceProp prop{}; char bus[32]{};
    if(cudaGetDeviceProperties(&prop,ordinal)!=cudaSuccess || cudaDeviceGetPCIBusId(bus,sizeof(bus),ordinal)!=cudaSuccess)
      throw std::runtime_error("cannot inspect target CUDA device");
    if(lowercase(bus)!=lowercase(wanted_bus[ordinal]) || std::string(prop.name).find("GV100")==std::string::npos)
      throw std::runtime_error("target CUDA ordinal resolves to a non-GV100 physical device");
    std::cerr<<"TARGET_GPU_MAPPING logical_ordinal="<<ordinal<<" pci_bus="<<bus<<" name="<<prop.name<<std::endl;
  }
  *devices={0,1};
}
static std::string gpu_uuid(int device) {
  char bus[32]{}; if (cudaDeviceGetPCIBusId(bus, sizeof(bus), device) != cudaSuccess) return "unavailable";
  std::string bus_key(bus); for(char& ch:bus_key)ch=static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  const auto separator=bus_key.find(':'); const std::string bus_tail=separator==std::string::npos?bus_key:bus_key.substr(separator+1);
  FILE* pipe = popen("nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader", "r");
  if (!pipe) return "unavailable"; char line[256]{}; std::string result="unavailable";
  while (fgets(line, sizeof(line), pipe)) { std::string row(line), lower=row; for(char& ch:lower)ch=static_cast<char>(std::tolower(static_cast<unsigned char>(ch))); if (lower.find(bus_key) != std::string::npos || lower.find(bus_tail) != std::string::npos) { const auto comma=row.find(','); if(comma!=std::string::npos){result=row.substr(comma+2); while(!result.empty()&&(result.back()=='\n'||result.back()=='\r'))result.pop_back();} break; } }
  pclose(pipe); return result;
}
static GlobalParameters initial(double beta, int nu2, int knots) {
  Matrix gamma(5,1), mu(knots,1), sigma(knots,knots); const double values[] = {-1,2,3,-2,1};
  for (int i=0;i<5;++i) gamma(i,0)=values[i]; return {gamma,mu,sigma,.25,1.,beta,nu2};
}
static void row(std::ofstream& out, const std::string& method, int iteration,
                const GlobalParameters& p, double step, int halves) {
  out << method << ',' << iteration;
  for (int i=0;i<5;++i) out << ',' << std::setprecision(17) << p.gamma(i,0);
  out << ',' << p.delta << ',' << p.alpha << ',' << p.range << ',' << step << ',' << halves << '\n';
}
int main(int argc, char** argv) {
  try {
    fs::path fixture_root, output; int scenario=-1, replication=-1, workers_count=10, rounds=50; bool p_only=false, two_worker_target=false, two_worker_2k_timing=false, two_worker_2k_m400=false, two_worker_2k_m400_smoke=false, two_worker_2k_m400_production=false, two_worker_paper_production=false, two_worker_2k_m200_production=false, one_round_timing=false; std::string devices="0";
    for (int i=1;i<argc;++i) { std::string flag=argv[i];
      if(flag=="--p-only") { p_only=true; continue; }
      if(flag=="--two-worker-15k-target") { two_worker_target=true; continue; }
      if(flag=="--two-worker-2k-timing-target") { two_worker_2k_timing=true; continue; }
      if(flag=="--two-worker-2k-m400-target") { two_worker_2k_m400=true; continue; }
      if(flag=="--two-worker-2k-m400-smoke") { two_worker_2k_m400_smoke=true; continue; }
      if(flag=="--two-worker-2k-m400-production") { two_worker_2k_m400_production=true; continue; }
      if(flag=="--two-worker-paper-production") { two_worker_paper_production=true; continue; }
      if(flag=="--two-worker-2k-m200-production") { two_worker_2k_m200_production=true; continue; }
      if(flag=="--one-round-timing") { one_round_timing=true; continue; }
      if (i+1>=argc) throw std::runtime_error("missing value for "+flag);
      if(flag=="--fixture-root") fixture_root=argv[++i]; else if(flag=="--output-dir") output=argv[++i];
      else if(flag=="--scenario-id") scenario=std::stoi(argv[++i]); else if(flag=="--replication") replication=std::stoi(argv[++i]);
      else if(flag=="--worker-devices") devices=argv[++i]; else if(flag=="--workers") workers_count=std::stoi(argv[++i]); else if(flag=="--rounds") rounds=std::stoi(argv[++i]); else throw std::runtime_error("unknown flag "+flag); }
    if(fixture_root.empty()||output.empty()||scenario<0||scenario>8||replication<0) throw std::runtime_error("require --fixture-root --output-dir --scenario-id --replication");
    const std::array<int,9> nu2{1,1,1,3,3,3,5,5,5}; const std::array<double,9> beta{.033,.1,.234,.037,.11,.256,.038,.113,.264};
    const std::string scenario_name = "scenario_" + std::string(scenario < 10 ? "0" : "") + std::to_string(scenario);
    const std::string replication_name = "rep_" + std::string(replication < 10 ? "00" : replication < 100 ? "0" : "") + std::to_string(replication);
    const fs::path fixture_dir=fixture_root/scenario_name/replication_name;
    if (workers_count <= 0 || rounds <= 0) throw std::runtime_error("workers and rounds must be positive");
    Figure1Fixture fixture; load_figure1_python_fixture(fixture_dir,&fixture); auto workers=partition_figure1_fixture(fixture,workers_count);
    auto worker_devices=parse_devices(devices); while(worker_devices.size()<workers.size()) worker_devices.push_back(worker_devices.back()); if(worker_devices.size()!=workers.size()) throw std::runtime_error("provide one device per worker or a prefix to repeat");
    const std::vector<int> requested_worker_devices=worker_devices;
    if (int(two_worker_target)+int(two_worker_2k_timing)+int(two_worker_2k_m400)+int(two_worker_2k_m400_smoke)+int(two_worker_2k_m400_production)+int(two_worker_paper_production)+int(two_worker_2k_m200_production) > 1) throw std::runtime_error("choose exactly one two-worker target");
    if (two_worker_target || two_worker_2k_timing || two_worker_2k_m400 || two_worker_2k_m400_smoke || two_worker_2k_m400_production || two_worker_paper_production || two_worker_2k_m200_production) require_target_gv100_mapping(&worker_devices);
    int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess) throw std::runtime_error("CUDA unavailable"); for(int device:worker_devices)if(device<0||device>=count)throw std::runtime_error("invalid CUDA device");
    std::ifstream fixture_metadata_stream(fixture_dir/"metadata.json");
    if (!fixture_metadata_stream) throw std::runtime_error("missing fixture metadata");
    const std::string fixture_metadata((std::istreambuf_iterator<char>(fixture_metadata_stream)), std::istreambuf_iterator<char>());
    if (two_worker_target) {
      using Target = NWorker15kFigure1Config;
      if (!p_only || scenario != Target::scenario_id || replication != Target::replication ||
          workers_count != Target::workers || rounds != (one_round_timing ? 1 : Target::rounds) ||
          requested_worker_devices.size() != Target::cuda_devices.size() ||
          requested_worker_devices[0] != Target::cuda_devices[0] || requested_worker_devices[1] != Target::cuda_devices[1])
        throw std::runtime_error("two-worker 15k target requires P-only scenario 4/replication 0, 50 rounds, workers/devices=2/1,2");
      if (fixture.sites.rows != Target::total_n || fixture.knots.rows != Target::knots || fixture.x.cols != Target::covariates ||
          workers[0].sites.rows != Target::local_n || workers[1].sites.rows != Target::local_n)
        throw std::runtime_error("two-worker 15k target requires one global 30000-row fixture partitioned into consecutive 15000-row workers");
      for (const char* required : {"\"scenario_id\": 4", "\"replication\": 0", "\"seed\": 1", "\"n\": 30000", "\"workers\": 2", "\"observations_per_worker\": 15000", "\"knots\": 2000", "\"partition_order\": \"global_then_consecutive\""})
        if (fixture_metadata.find(required) == std::string::npos) throw std::runtime_error("two-worker 15k target fixture metadata mismatch");
    }
    if (two_worker_2k_timing) {
      using Target = NWorker2kFigure1TimingConfig;
      if (!p_only || scenario != Target::scenario_id || replication != Target::replication || workers_count != Target::workers || rounds != 1 ||
          requested_worker_devices.size()!=2 || requested_worker_devices[0]!=1 || requested_worker_devices[1]!=2)
        throw std::runtime_error("two-worker 2k timing target requires P-only scenario 4/replication 0, one round, workers/devices=2/1,2");
      if (fixture.sites.rows != Target::total_n || fixture.knots.rows != Target::knots || fixture.x.cols != Target::covariates ||
          workers[0].sites.rows != Target::local_n || workers[1].sites.rows != Target::local_n)
        throw std::runtime_error("two-worker 2k timing target requires one global 4000-row fixture partitioned into consecutive 2000-row workers");
      for (const char* required : {"\"scenario_id\": 4", "\"replication\": 0", "\"seed\": 1", "\"n\": 4000", "\"workers\": 2", "\"observations_per_worker\": 2000", "\"knots\": 2000", "\"partition_order\": \"global_then_consecutive\""})
        if (fixture_metadata.find(required) == std::string::npos) throw std::runtime_error("two-worker 2k timing target fixture metadata mismatch");
    }
    if (two_worker_2k_m400) {
      using Target = NWorker2kM400Figure1Config;
      if (!p_only || scenario != Target::scenario_id || replication != Target::replication || workers_count != Target::workers || rounds != 1 ||
          requested_worker_devices.size()!=2 || requested_worker_devices[0]!=1 || requested_worker_devices[1]!=2)
        throw std::runtime_error("two-worker 2k m400 target requires P-only scenario 4/replication 0, one round, workers/devices=2/1,2");
      if (fixture.sites.rows != Target::total_n || fixture.knots.rows != Target::knots || fixture.x.cols != Target::covariates ||
          workers[0].sites.rows != Target::local_n || workers[1].sites.rows != Target::local_n)
        throw std::runtime_error("two-worker 2k m400 target requires one global 4000-row fixture partitioned into consecutive 2000-row workers");
      for (const char* required : {"\"scenario_id\": 4", "\"replication\": 0", "\"seed\": 1", "\"n\": 4000", "\"workers\": 2", "\"observations_per_worker\": 2000", "\"knots\": 400", "\"partition_order\": \"global_then_consecutive\""})
        if (fixture_metadata.find(required) == std::string::npos) throw std::runtime_error("two-worker 2k m400 target fixture metadata mismatch");
    }
    if (two_worker_2k_m400_smoke) {
      using Target = NWorker2kM400Figure1SmokeConfig;
      if (!p_only || scenario != Target::scenario_id || replication != Target::replication || workers_count != Target::workers || rounds != Target::rounds ||
          requested_worker_devices.size()!=2 || requested_worker_devices[0]!=1 || requested_worker_devices[1]!=2)
        throw std::runtime_error("two-worker 2k m400 smoke requires P-only scenario 4/replication 0, 50 rounds, workers/devices=2/1,2");
      if (fixture.sites.rows != Target::total_n || fixture.knots.rows != Target::knots || fixture.x.cols != Target::covariates ||
          workers[0].sites.rows != Target::local_n || workers[1].sites.rows != Target::local_n)
        throw std::runtime_error("two-worker 2k m400 smoke requires one global 4000-row fixture partitioned into consecutive 2000-row workers");
      for (const char* required : {"\"scenario_id\": 4", "\"replication\": 0", "\"seed\": 1", "\"n\": 4000", "\"workers\": 2", "\"observations_per_worker\": 2000", "\"knots\": 400", "\"partition_order\": \"global_then_consecutive\""})
        if (fixture_metadata.find(required) == std::string::npos) throw std::runtime_error("two-worker 2k m400 smoke fixture metadata mismatch");
    }
    if (two_worker_2k_m400_production) {
      using Target = NWorker2kM400Figure1SmokeConfig;
      if (!p_only || workers_count != Target::workers || rounds != Target::rounds ||
          requested_worker_devices.size()!=2 || requested_worker_devices[0]!=1 || requested_worker_devices[1]!=2)
        throw std::runtime_error("two-worker 2k m400 production requires P-only, 50 rounds, workers/devices=2/1,2");
      if (fixture.sites.rows != Target::total_n || fixture.knots.rows != Target::knots || fixture.x.cols != Target::covariates ||
          workers[0].sites.rows != Target::local_n || workers[1].sites.rows != Target::local_n)
        throw std::runtime_error("two-worker 2k m400 production requires one global n=4000, m=400 fixture partitioned into consecutive 2000-row workers");
      for (const char* required : {"\"n\": 4000", "\"p\": 5", "\"knots\": 400", "\"workers\": 2", "\"observations_per_worker\": 2000", "\"partition_order\": \"global_then_consecutive\""})
        if (fixture_metadata.find(required) == std::string::npos) throw std::runtime_error("two-worker 2k m400 production fixture metadata mismatch");
    }
    if (two_worker_paper_production) {
      if (!p_only || workers_count!=2 || rounds<1 || requested_worker_devices.size()!=2 ||
          requested_worker_devices[0]!=1 || requested_worker_devices[1]!=2)
        throw std::runtime_error("two-worker paper production requires P-only workers/devices=2/1,2");
      if (fixture.sites.rows!=1000 || fixture.knots.rows!=100 || fixture.x.cols!=5 ||
          workers[0].sites.rows!=500 || workers[1].sites.rows!=500)
        throw std::runtime_error("two-worker paper production requires global n=1000, m=100, q=5, consecutive 500-row workers");
      for (const char* required : {"\"n\": 1000", "\"p\": 5", "\"knots\": 100", "\"workers\": 2", "\"observations_per_worker\": 500", "\"partition_order\": \"global_then_consecutive\""})
        if (fixture_metadata.find(required)==std::string::npos) throw std::runtime_error("two-worker paper production fixture metadata mismatch");
    }
    if (two_worker_2k_m200_production) {
      if (!p_only || workers_count!=2 || rounds!=50 || requested_worker_devices.size()!=2 ||
          requested_worker_devices[0]!=1 || requested_worker_devices[1]!=2)
        throw std::runtime_error("two-worker 2k m200 production requires P-only, 50 rounds, workers/devices=2/1,2");
      if (fixture.sites.rows!=2000 || fixture.knots.rows!=200 || fixture.x.cols!=5 ||
          workers[0].sites.rows!=1000 || workers[1].sites.rows!=1000)
        throw std::runtime_error("two-worker 2k m200 production requires one global n=2000, m=200 fixture partitioned into consecutive 1000-row workers");
      for (const char* required : {"\"n\": 2000", "\"p\": 5", "\"knots\": 200", "\"workers\": 2", "\"observations_per_worker\": 1000", "\"partition_order\": \"global_then_consecutive\""})
        if (fixture_metadata.find(required)==std::string::npos) throw std::runtime_error("two-worker 2k m200 production fixture metadata mismatch");
    }
    fs::create_directories(output); std::ofstream trajectory(output/"native_trajectory.csv"); trajectory<<"method,iteration,gamma0,gamma1,gamma2,gamma3,gamma4,delta,alpha,beta,accepted_step,halving_count\n"<<std::flush; std::ofstream timing(output/"native_round_timing.csv"); timing<<"iteration,mu_sigma_local_seconds,gamma_local_seconds,delta_theta_local_seconds,aggregation_seconds,global_update_seconds,broadcast_seconds,total_seconds\n"<<std::flush;
    std::ofstream manifest(output/"native_manifest.json"); manifest<<"{\n  \"provenance\": \"fresh_python_generated_raw_fixture\",\n  \"fixture_directory\": \""<<fixture_dir.string()<<"\",\n  \"fixture_metadata\": "<<fixture_metadata<<",\n  \"scenario_id\": "<<scenario<<",\n  \"replication\": "<<replication<<",\n  \"seed\": "<<(replication+1)<<",\n  \"nu2\": "<<nu2[scenario]<<",\n  \"beta\": "<<beta[scenario]<<",\n  \"alpha\": 1.0,\n  \"delta\": 0.25,\n  \"gamma\": [-1,2,3,-2,1],\n  \"n\": "<<fixture.sites.rows<<",\n  \"p\": "<<fixture.x.cols<<",\n  \"knots\": "<<fixture.knots.rows<<",\n  \"iterations\": "<<rounds<<",\n  \"convergence_rule\": \"fixed_rounds\",\n  \"workers\": "<<workers_count<<",\n  \"worker_rows\": "<<workers.front().sites.rows<<",\n  \"p_only\": "<<(p_only?"true":"false")<<",\n  \"two_worker_15k_target\": "<<(two_worker_target?"true":"false")<<",\n  \"worker_cuda_devices\": \""<<devices<<"\",\n  \"primary_cuda_uuid\": \""<<gpu_uuid(worker_devices.front())<<"\",\n  \"executable\": \"sync_gpu_figure1_fresh_case\",\n  \"toolchain\": \"CUDA 12.4 FP64 sm_70\",\n  \"git\": \"unavailable: workspace is not a usable git repository\"\n}\n";
    auto p=initial(beta[scenario],nu2[scenario],fixture.knots.rows); NWorkerPSynchronousExecutor p_executor(worker_devices); auto start=std::chrono::steady_clock::now();
    for(int iteration=0;iteration<rounds;++iteration){auto result=p_executor.run_round_with_checkpoints(workers,fixture.knots,p);row(trajectory,"P",iteration,p,result.after_delta_theta.accepted_step,result.after_delta_theta.halving_count); trajectory.flush(); const auto&t=result.timing; timing<<iteration<<','<<t.mu_sigma_local_seconds<<','<<t.gamma_local_seconds<<','<<t.delta_theta_local_seconds<<','<<t.aggregation_seconds<<','<<t.global_update_seconds<<','<<t.broadcast_seconds<<','<<t.total_seconds<<'\n'<<std::flush; p=result.after_delta_theta.parameters;} row(trajectory,"P",rounds,p,0.,0); trajectory.flush(); const double p_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if (p_only) { std::ofstream estimates(output/"native_estimates.csv"); estimates<<"method,gamma0,gamma1,gamma2,gamma3,gamma4,delta,alpha,beta,runtime_seconds,status\nP"; for(int i=0;i<5;++i) estimates<<','<<std::setprecision(17)<<p.gamma(i,0); estimates<<','<<p.delta<<','<<p.alpha<<','<<p.range<<','<<p_seconds<<",OK\n"; std::cout<<std::setprecision(17)<<"FRESH_CASE_P_SECONDS="<<p_seconds<<"\nFRESH_CASE_STATUS=PASS\n"; return 0; }
    GpuContext ind_context(worker_devices.front()); auto ind=initial(beta[scenario],nu2[scenario],fixture.knots.rows); start=std::chrono::steady_clock::now();
    for(int iteration=0;iteration<rounds;++iteration){auto result=run_ind_synchronous_round(ind_context,workers,ind);row(trajectory,"IND",iteration,ind,result.accepted_step,result.halving_count);ind=result.parameters;} row(trajectory,"IND",rounds,ind,0.,0); const double ind_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream estimates(output/"native_estimates.csv"); estimates<<"method,gamma0,gamma1,gamma2,gamma3,gamma4,delta,alpha,beta,runtime_seconds,status\n";
    for(const auto& item:std::vector<std::pair<std::string,GlobalParameters>>{{"P",p},{"IND",ind}}){estimates<<item.first;for(int i=0;i<5;++i)estimates<<','<<std::setprecision(17)<<item.second.gamma(i,0);estimates<<','<<item.second.delta<<','<<item.second.alpha<<','<<item.second.range<<','<<(item.first=="P"?p_seconds:ind_seconds)<<",OK\n";}
    std::cout<<std::setprecision(17)<<"FRESH_CASE_P_SECONDS="<<p_seconds<<"\nFRESH_CASE_IND_SECONDS="<<ind_seconds<<"\nFRESH_CASE_STATUS=PASS\n";
    return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 2; }
}
