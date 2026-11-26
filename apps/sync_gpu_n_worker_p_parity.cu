#include "fedspatial/gpu_context.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <typeinfo>

using namespace fedspatial;
namespace fs = std::filesystem;
struct Row { int round{}; std::string stage; std::vector<double> v; };
struct Error { double abs{}, rel{}; void add(double a, double b) { abs=std::max(abs,std::abs(a-b)); rel=std::max(rel,std::abs(a-b)/std::max(1.,std::abs(b))); } };
struct Errors { Error gamma,mu,sigma,delta,alpha,range; };

static Matrix load(const fs::path& p) { Matrix x; if (!read_matrix(p.string(), &x)) throw std::runtime_error("missing " + p.string()); return x; }
static std::string exception_details(const std::exception& error) { std::ostringstream out; out << "type=" << typeid(error).name() << " what=" << error.what(); if (const auto* system=dynamic_cast<const std::system_error*>(&error)) out << " code=" << system->code().value() << " category=" << system->code().category().name() << " message=" << system->code().message(); return out.str(); }
static std::vector<Row> checkpoints(int n) {
  std::ifstream in("sync_gpu/tests/fixtures/sync_gpu/N_WORKER_" + std::to_string(n) + "_P_CHECKPOINTS.csv");
  if (!in) throw std::runtime_error("missing N-worker Python fixture; run export_python_fixture.py");
  std::string line, field; std::getline(in,line); std::vector<Row> rows;
  while (std::getline(in,line)) { std::stringstream s(line); Row r; std::getline(s,field,','); r.round=std::stoi(field); std::getline(s,r.stage,','); while(std::getline(s,field,',')) r.v.push_back(std::stod(field)); if(r.v.size()!=12) throw std::runtime_error("bad checkpoint row"); rows.push_back(std::move(r)); }
  return rows;
}
static WorkerInput worker(int id) {
  Matrix d=load("sync_gpu/tests/fixtures/sync_gpu/N_WORKER_"+std::to_string(id)+".txt");
  WorkerInput w{Matrix(d.rows,2),Matrix(d.rows,1),Matrix(d.rows,d.cols-3)};
  for(int i=0;i<d.rows;++i){w.sites(i,0)=d(i,0);w.sites(i,1)=d(i,1);w.z(i,0)=d(i,2);for(int j=0;j<w.x.cols;++j)w.x(i,j)=d(i,j+3);} return w;
}
static Matrix knots(){ Matrix k(2,2); k(0,0)=k(0,1)=0.; k(1,0)=k(1,1)=1.5; return k; }
static GlobalParameters initial(){ Matrix g(1,1),m(2,1),s(2,2);g(0,0)=.15;m(0,0)=.2;m(1,0)=-.1;s(0,0)=.7;s(0,1)=s(1,0)=.05;s(1,1)=.4;return {g,m,s,1.7,1.2,.8,3}; }
static void compare(const GlobalParameters& x,const Row&r,Errors&e){e.gamma.add(x.gamma(0,0),r.v[0]);e.mu.add(x.mu(0,0),r.v[1]);e.mu.add(x.mu(1,0),r.v[2]);e.sigma.add(x.sigma(0,0),r.v[3]);e.sigma.add(x.sigma(0,1),r.v[4]);e.sigma.add(x.sigma(1,0),r.v[5]);e.sigma.add(x.sigma(1,1),r.v[6]);e.delta.add(x.delta,r.v[7]);e.alpha.add(x.alpha,r.v[8]);e.range.add(x.range,r.v[9]);}
static bool same(const std::vector<PWorkerRoundCheckpoints>&a,const std::vector<PWorkerRoundCheckpoints>&b){if(a.size()!=b.size())return false;for(size_t r=0;r<a.size();++r)for(const GlobalParameters* x:{&a[r].after_mu_sigma,&a[r].after_gamma,&a[r].after_delta_theta.parameters}){const GlobalParameters*y=x==&a[r].after_mu_sigma?&b[r].after_mu_sigma:x==&a[r].after_gamma?&b[r].after_gamma:&b[r].after_delta_theta.parameters;if(x->gamma.a!=y->gamma.a||x->mu.a!=y->mu.a||x->sigma.a!=y->sigma.a||std::memcmp(&x->delta,&y->delta,sizeof(double))||std::memcmp(&x->alpha,&y->alpha,sizeof(double))||std::memcmp(&x->range,&y->range,sizeof(double)))return false;}return true;}
// The fixture is m=2, so this is an exact focused SPD/symmetry audit of the
// compact checkpoint downloaded from the persistent central device state.
static bool spd_symmetric(const Matrix&s){if(s.rows!=s.cols)return false;double sym=0.;for(int i=0;i<s.rows;++i)for(int j=0;j<s.cols;++j)sym=std::max(sym,std::abs(s(i,j)-s(j,i)));if(sym!=0.)return false;double l00=std::sqrt(s(0,0));if(!(l00>0.))return false;double l10=s(1,0)/l00;double l11sq=s(1,1)-l10*l10;return l11sq>0.;}
static std::vector<PWorkerRoundCheckpoints> run(int n,const std::vector<int>&devices){const char* text=std::getenv("NWORKER_TRAJECTORY_ROUNDS");const int rounds=text?std::max(2,std::stoi(text)):2;NWorkerPSynchronousExecutor ex(devices);std::vector<WorkerInput>w;for(int i=0;i<n;++i)w.push_back(worker(i));std::vector<PWorkerRoundCheckpoints>out;auto state=initial();auto k=knots();for(int r=0;r<rounds;++r){auto v=ex.run_round_with_checkpoints(w,k,state);state=v.after_delta_theta.parameters;out.push_back(std::move(v));}return out;}
int main(int argc,char**argv){try{int n=argc>1?std::stoi(argv[1]):10;if(n<1||n>10)throw std::runtime_error("J must be 1..10");int available=0;if(cudaGetDeviceCount(&available)!=cudaSuccess)return 77;std::vector<int>d;for(int i=2;i<argc;++i)d.push_back(std::stoi(argv[i]));if(d.empty())d.push_back(0);while((int)d.size()<n)d.push_back(d.back());if((int)d.size()!=n)throw std::runtime_error("provide one device per worker, or a prefix to repeat");for(int v:d)if(v<0||v>=available)return 77;auto expected=checkpoints(n);if(expected.size()!=6)throw std::runtime_error("expected two rounds x three stages");auto first=run(n,d),second=run(n,d);Errors e{};int q=0;for(int r=0;r<2;++r)for(const char*stage:{"mu_sigma","gamma","delta_theta"}){if(expected[q].round!=r||expected[q].stage!=stage)throw std::runtime_error("checkpoint ordering mismatch");const auto& state=std::string(stage)=="mu_sigma"?first[r].after_mu_sigma:std::string(stage)=="gamma"?first[r].after_gamma:first[r].after_delta_theta.parameters;compare(state,expected[q],e);if(std::string(stage)=="delta_theta"&&(std::abs(first[r].after_delta_theta.accepted_step-expected[q].v[10])>1e-15||first[r].after_delta_theta.halving_count!=(int)expected[q].v[11]))throw std::runtime_error("Newton step/halving mismatch");++q;}bool deterministic=same(first,second),raw=true,spd=true;long long peer=0,pinned=0,model_h2d=0,model_d2h=0,host_compute_d2h=0;for(const auto&v:first){raw=raw&&v.timing.raw_data_stationary;spd=spd&&spd_symmetric(v.after_mu_sigma.sigma);peer+=v.timing.peer_summary_bytes;pinned+=v.timing.pinned_summary_bytes;model_h2d+=v.timing.model_h2d_bytes;model_d2h+=v.timing.model_d2h_bytes;host_compute_d2h+=v.timing.host_compute_d2h_bytes;}double maximum=std::max({e.gamma.abs,e.mu.abs,e.sigma.abs,e.delta.abs,e.alpha.abs,e.range.abs});auto print=[](const char*n,const Error&e){std::cout<<n<<"_MAX_ABS="<<e.abs<<" MAX_REL="<<e.rel<<'\n';};std::cout<<std::setprecision(17)<<"N_WORKER_J="<<n<<"\n";print("GAMMA",e.gamma);print("MU",e.mu);print("SIGMA",e.sigma);print("DELTA",e.delta);print("ALPHA",e.alpha);print("RANGE",e.range);std::cout<<"N_WORKER_DETERMINISM="<<(deterministic?"BYTE_IDENTICAL":"FAIL")<<'\n'<<"N_WORKER_SPD_SYMMETRY="<<(spd?"PASS":"FAIL")<<'\n'<<"N_WORKER_P2P_SUMMARY_BYTES="<<peer<<'\n'<<"N_WORKER_PINNED_FALLBACK_SUMMARY_BYTES="<<pinned<<'\n'<<"N_WORKER_MODEL_H2D_BYTES="<<model_h2d<<'\n'<<"N_WORKER_MODEL_D2H_BYTES="<<model_d2h<<'\n'<<"N_WORKER_HOST_COMPUTE_D2H_BYTES="<<host_compute_d2h<<'\n'<<"N_WORKER_RAW_DATA_STATIONARY="<<(raw?"PASS":"FAIL")<<'\n'<<"N_WORKER_P_PARITY="<<(maximum<=1e-10&&deterministic&&raw&&spd?"PASS":"FAIL")<<'\n';return maximum<=1e-10&&deterministic&&raw&&spd?0:1;}catch(const std::exception&x){std::cerr<<"N_WORKER_TOP_LEVEL_FAILURE "<<exception_details(x)<<'\n';return 2;}}
