#include "fedspatial/cpu_backend.hpp"
#include "fedspatial/figure1_fixture.hpp"
#include "fedspatial/multi_gpu_worker_group.hpp"
#ifdef FEDSPATIAL_RUNTIME_CUDA
#include "fedspatial/gpu_context.hpp"
#endif
#include <mpi.h>
#include <omp.h>
#ifdef FEDSPATIAL_MKL_LOCAL
#include <mkl.h>
#endif
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <sys/resource.h>

using namespace fedspatial;
namespace fs=std::filesystem;
namespace {
struct Options {
  std::string backend="cpu";
  fs::path fixture,output;
  int workers=0,rounds=50,ranks=1,threads=1,nu2=3;
  double delta=.25,alpha=1.,range=.11;
  bool small=false,text=false,zero_mean=false;
};
int positive_int(const std::string& s) {
  size_t used=0;long long n=std::stoll(s,&used);
  if(used!=s.size()||n<=0||n>std::numeric_limits<int>::max())throw std::runtime_error("expected a positive integer: "+s);
  return int(n);
}
Options options(int argc,char**argv) {
  Options o;
  for(int i=1;i<argc;++i) {
    std::string key=argv[i],v;
    if(key=="--small-fixture"){o.small=true;continue;}
    if(key=="--text-fixture"){o.text=true;continue;}
    if(key=="--zero-mean"){o.zero_mean=true;continue;}
    auto eq=key.find('=');
    if(eq!=std::string::npos){v=key.substr(eq+1);key.resize(eq);}
    else{if(++i==argc)throw std::runtime_error("missing value for "+key);v=argv[i];}
    if(key=="--backend")o.backend=v;
    else if(key=="--fixture-dir")o.fixture=v;
    else if(key=="--output-dir")o.output=v;
    else if(key=="--workers")o.workers=positive_int(v);
    else if(key=="--rounds")o.rounds=positive_int(v);
    else if(key=="--ranks-per-worker")o.ranks=positive_int(v);
    else if(key=="--cpus-per-rank")o.threads=positive_int(v);
    else if(key=="--nu2")o.nu2=positive_int(v);
    else if(key=="--init-delta")o.delta=std::stod(v);
    else if(key=="--init-alpha")o.alpha=std::stod(v);
    else if(key=="--init-range"||key=="--range")o.range=std::stod(v);
    else throw std::runtime_error("unknown option "+key);
  }
  if(o.backend!="cpu"&&o.backend!="gpu")throw std::runtime_error("backend must be cpu or gpu");
#ifndef FEDSPATIAL_RUNTIME_CUDA
  if(o.backend=="gpu")throw std::runtime_error("GPU backend unavailable in this CPU-only build");
#endif
  if(o.backend=="gpu"&&o.ranks!=1)throw std::runtime_error("GPU backend requires ranks-per-worker=1; use CPU for hierarchical ranks");
  if(o.fixture.empty()||o.output.empty()||o.workers<=0)throw std::runtime_error("require --fixture-dir, --output-dir and --workers");
  if(o.nu2!=1&&o.nu2!=3&&o.nu2!=5)throw std::runtime_error("nu2 must be 1, 3 or 5");
  for(double x:{o.delta,o.alpha,o.range})if(!(x>0)||!std::isfinite(x))throw std::runtime_error("initial covariance parameters must be finite and positive");
  if(o.small&&(o.text||o.nu2!=3))throw std::runtime_error("small reference fixture requires nu2=3 and excludes text-fixture");
  return o;
}
Matrix load(const fs::path& p) {Matrix a;if(!read_matrix(p.string(),&a))throw std::runtime_error("cannot load "+p.string());return a;}
WorkerInput unpack_worker(const Matrix& d) {
  if(d.rows<=0||d.cols<4)throw std::runtime_error("invalid worker fixture");
  WorkerInput w{Matrix(d.rows,2),Matrix(d.rows,1),Matrix(d.rows,d.cols-3)};
  for(int i=0;i<d.rows;++i){for(int j=0;j<2;++j)w.sites(i,j)=d(i,j);w.z(i,0)=d(i,2);for(int j=0;j<w.x.cols;++j)w.x(i,j)=d(i,j+3);}
  return w;
}
std::vector<double> pack(const GlobalParameters& p) {
  auto v=p.gamma.a;v.insert(v.end(),p.mu.a.begin(),p.mu.a.end());v.insert(v.end(),p.sigma.a.begin(),p.sigma.a.end());
  v.insert(v.end(),{p.delta,p.alpha,p.range});return v;
}
void broadcast(GlobalParameters& p,const MultiGpuWorkerGroup& group) {
  auto v=pack(p);
  if(v.size()>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("model exceeds MPI count limit");
  if(group.is_leader) {
    MPI_Bcast(v.data(),int(v.size()),MPI_DOUBLE,0,group.leaders_comm);
    if(group.leader_rank==0)cpu_add_communication_bytes(
        std::uint64_t(v.size())*sizeof(double)*(group.worker_count-1));
  }
  MPI_Bcast(v.data(),int(v.size()),MPI_DOUBLE,0,group.worker_comm);
  if(group.worker_rank==0)cpu_add_communication_bytes(
      std::uint64_t(v.size())*sizeof(double)*(group.worker_size-1));
  size_t k=0;for(double& x:p.gamma.a)x=v[k++];for(double& x:p.mu.a)x=v[k++];for(double& x:p.sigma.a)x=v[k++];
  p.delta=v[k++];p.alpha=v[k++];p.range=v[k++];
}
std::vector<double> ordered_summary(const std::vector<double>& local,const MultiGpuWorkerGroup& g,bool mean) {
  if(!g.is_leader)return {};
  if(local.size()>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("summary exceeds MPI count limit");
  std::vector<double> slots(g.leader_rank==0?local.size()*g.worker_count:0),total(local.size());
  MPI_Gather(local.data(),int(local.size()),MPI_DOUBLE,slots.data(),int(local.size()),MPI_DOUBLE,0,g.leaders_comm);
  if(g.leader_rank==0)cpu_add_communication_bytes(
      std::uint64_t(local.size())*sizeof(double)*(g.worker_count-1));
  if(g.leader_rank==0) {
    for(int w=0;w<g.worker_count;++w)for(size_t i=0;i<local.size();++i)total[i]+=slots[w*local.size()+i];
    if(mean)for(double& x:total)x/=g.worker_count;
  }
  return total;
}
void row(std::ofstream& out,int r,const char* stage,const GlobalParameters& p,double step=1,int halves=0) {
  out<<r<<','<<stage<<std::setprecision(17);for(double x:pack(p))out<<','<<x;out<<','<<step<<','<<halves<<'\n';out.flush();
  if(!out)throw std::runtime_error("checkpoint write failed");
}
void local_mode(const Options& o,const MultiGpuWorkerGroup& g,bool on) {
#ifdef FEDSPATIAL_SCALAPACK
  // The service owns intra-worker communication; scalar functions run on leader.
  if(o.backend=="cpu"){cpu_set_worker_comm(0,1,nullptr);return;}
#endif
  if(o.backend=="cpu")cpu_set_worker_comm(on?g.worker_rank:0,on?g.worker_size:1,on?&g.worker_comm:nullptr);
}
void cpu_execute(const MultiGpuWorkerGroup& g,const std::function<void()>& f) {
#ifdef FEDSPATIAL_SCALAPACK
  cpu_worker_execute(&g.worker_comm,f);
#else
  f();
#endif
}
}
int main(int argc,char**argv) {
  int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
  int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  try {
    if(provided<MPI_THREAD_FUNNELED)throw std::runtime_error("MPI lacks FUNNELED thread support");
    if(argc==2&&std::string(argv[1])=="--help") {
      if(rank==0)std::cout<<"sync_mpi_trajectory --backend=cpu|gpu --workers=N --ranks-per-worker=N --cpus-per-rank=N\n"
        <<"  --fixture-dir=PATH --output-dir=PATH [--rounds=50] [--nu2=1|3|5]\n"
        <<"  [--init-delta=0.25 --init-alpha=1 --init-range=0.11]\n"
        <<"  [--small-fixture | --text-fixture] [--zero-mean]\n";
      MPI_Finalize();return 0;
    }
    const auto o=options(argc,argv);omp_set_dynamic(0);omp_set_num_threads(o.threads);
#ifdef FEDSPATIAL_MKL_LOCAL
    // Make --cpus-per-rank authoritative for both OpenMP covariance loops and
    // oneMKL BLAS/LAPACK kernels.  Environment defaults must not silently
    // oversubscribe a multi-rank worker.
    mkl_set_dynamic(0);
    mkl_set_num_threads_local(o.threads);
#endif
    auto group=MultiGpuWorkerGroup::create(MPI_COMM_WORLD,o.ranks);
    if(group.worker_count!=o.workers)throw std::runtime_error("MPI size must equal workers * ranks-per-worker");
    Matrix knots;WorkerInput input;
    bool load_input=true;
#ifdef FEDSPATIAL_SCALAPACK
    if(o.backend=="cpu")load_input=group.is_leader;
#endif
    if(load_input) {
    if(o.small) {
      input=unpack_worker(load(o.fixture/("N_WORKER_"+std::to_string(group.worker_id)+".txt")));
      knots=Matrix(2,2);knots(1,0)=knots(1,1)=1.5;
    } else if(o.text) {
      input=unpack_worker(load(o.fixture/("worker_"+std::to_string(group.worker_id)+".txt")));
      knots=load(o.fixture/"knots.txt");
    } else {
      Figure1Fixture fixture;load_figure1_python_fixture(o.fixture,&fixture);
      input=partition_figure1_fixture(fixture,o.workers).at(group.worker_id);knots=fixture.knots;
    }
    }
#ifdef FEDSPATIAL_SCALAPACK
    if(o.backend=="cpu") {
      int dimensions[2]={input.x.cols,knots.rows};
      MPI_Bcast(dimensions,2,MPI_INT,0,group.worker_comm);
      if(!group.is_leader){input.x.cols=dimensions[0];knots=Matrix(dimensions[1],2);}
    }
#endif
    GlobalParameters state{Matrix(input.x.cols,1),Matrix(knots.rows,1),Matrix(knots.rows,knots.rows),o.delta,o.alpha,o.range,o.nu2};
    if(o.small) {
      if(input.x.cols!=1)throw std::runtime_error("small fixture must have one covariate");
      state.gamma(0,0)=.15;state.mu(0,0)=.2;state.mu(1,0)=-.1;
      state.sigma(0,0)=.7;state.sigma(0,1)=state.sigma(1,0)=.05;state.sigma(1,1)=.4;
      state.delta=1.7;state.alpha=1.2;state.range=.8;
    } else if(!o.zero_mean) {
      if(input.x.cols!=5)throw std::runtime_error("Figure-1 initial state requires five covariates");
      const double gamma[]={-1,2,3,-2,1};std::copy(gamma,gamma+5,state.gamma.a.begin());
    }
#ifdef FEDSPATIAL_RUNTIME_CUDA
    std::unique_ptr<GpuContext> gpu;
    if(o.backend=="gpu") {
      MPI_Comm shared;MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&shared);
      int local_rank=0;MPI_Comm_rank(shared,&local_rank);MPI_Comm_free(&shared);
      gpu=std::make_unique<GpuContext>(local_rank);
    }
#endif
    const auto initial_state=state;
    std::ofstream checkpoints,timing;
    if(rank==0) {
      fs::create_directories(o.output);checkpoints.open(o.output/"checkpoints.csv");timing.open(o.output/"timing.csv");
      if(!checkpoints||!timing)throw std::runtime_error("cannot create output files");
      checkpoints<<"round,stage";for(int i=0;i<state.gamma.rows;++i)checkpoints<<",gamma"<<i;
      for(int i=0;i<knots.rows;++i)checkpoints<<",mu"<<i;
      for(int i=0;i<knots.rows;++i)for(int j=0;j<knots.rows;++j)checkpoints<<",sigma"<<i<<'_'<<j;
      checkpoints<<",delta,alpha,range,accepted_step,halving_count\n";
      timing<<"round,seconds,mu_sigma_seconds,gamma_seconds,delta_theta_seconds,logical_communication_bytes\n";
    }
    std::cout<<"MPI_MAP rank="<<rank<<" worker="<<group.worker_id<<" worker_rank="<<group.worker_rank
             <<" backend="<<o.backend<<" cpus_per_rank="<<o.threads<<std::endl;
    cpu_reset_communication_bytes();
    std::uint64_t run_communication_bytes=0;
    for(int r=0;r<o.rounds;++r) {
      const std::uint64_t communication_begin=cpu_communication_bytes();
      const double start=MPI_Wtime();local_mode(o,group,true);
      MuSigmaLocalSummary lm;
      LocalComputationResult cpu_local_state;
      if(o.backend=="cpu")cpu_execute(group,[&]{
        cpu_local_state=cpu_local(input,knots,state,o.zero_mean);
        lm=cpu_local_state.summary;
      });

#ifdef FEDSPATIAL_RUNTIME_CUDA
      else lm=compute_mu_sigma_local(*gpu,input,knots,state).summary;
#endif
      const double mu_end=MPI_Wtime();
      local_mode(o,group,false);
      auto send=lm.mu.a;send.insert(send.end(),lm.sigma.a.begin(),lm.sigma.a.end());
      auto total=ordered_summary(send,group,true);
      if(rank==0) {
        int m=knots.rows;MuSigmaLocalSummary mean{Matrix(m,1),Matrix(m,m)};
        std::copy(total.begin(),total.begin()+m,mean.mu.a.begin());std::copy(total.begin()+m,total.end(),mean.sigma.a.begin());
        cpu_update_mu_sigma(state,knots,mean,o.workers);row(checkpoints,r,"mu_sigma",state);
      }
      broadcast(state,group);local_mode(o,group,true);
      const double gamma_start=MPI_Wtime();
      GammaLocalSummary lg;
      if(!o.zero_mean) {
        if(o.backend=="cpu")cpu_execute(group,[&]{
          lg=cpu_gamma_from_local(input,state,cpu_local_state);
        });
#ifdef FEDSPATIAL_RUNTIME_CUDA
        else lg=compute_gamma_local(*gpu,input,knots,state);
#endif
      }
      const double gamma_end=MPI_Wtime();
      local_mode(o,group,false);
      if(!o.zero_mean) {
        send=lg.gamma.a;send.insert(send.end(),lg.xx.a.begin(),lg.xx.a.end());total=ordered_summary(send,group,true);
        if(rank==0) {
          int q=state.gamma.rows;Matrix rhs(q,1),xx(q,q);
          std::copy(total.begin(),total.begin()+q,rhs.a.begin());std::copy(total.begin()+q,total.end(),xx.a.begin());
          state.gamma=solve_spd(xx,rhs);row(checkpoints,r,"gamma",state);
        }
        broadcast(state,group);
      }
      local_mode(o,group,true);
      if(o.backend=="cpu") {
        // The remaining block needs only B, M, chol(M), and logdet(M). Release
        // summaries and solved right-hand sides before derivative allocation.
        cpu_local_state.cstar=Matrix();
        // Retain the n-by-1 residual: Block 3 reuses it and the zero-mean
        // benchmark thereby avoids every X/gamma operation.
        cpu_local_state.rinv_B=Matrix();
        cpu_local_state.bt_rinv_B=Matrix();
        cpu_local_state.rinv_e=Matrix();
        cpu_local_state.bt_rinv_e=Matrix();
        cpu_local_state.summary={Matrix(),Matrix()};
      }
      const double delta_start=MPI_Wtime();
      DeltaThetaLocalResult lt;
      if(o.backend=="cpu")cpu_execute(group,[&]{
        lt=cpu_delta_theta_from_local(input,knots,state,cpu_local_state,o.zero_mean);
      });
#ifdef FEDSPATIAL_RUNTIME_CUDA
      else lt=compute_delta_theta_local(*gpu,input,knots,state);
#endif
      const double delta_end=MPI_Wtime();
      local_mode(o,group,false);send={lt.objective};send.insert(send.end(),lt.gradient.begin(),lt.gradient.end());send.insert(send.end(),lt.hessian.begin(),lt.hessian.end());
      total=ordered_summary(send,group,false);
      if(rank==0) {
        Matrix gradient(3,1),hessian(3,3),kg(3,1),kh(3,3);
        std::copy(total.begin()+1,total.begin()+4,gradient.a.begin());std::copy(total.begin()+4,total.end(),hessian.a.begin());
#ifdef FEDSPATIAL_RUNTIME_CUDA
        if(o.backend=="gpu")compute_legacy_global_k_correction_for_test(*gpu,knots,state,&kg,&kh);else
#endif
        cpu_k_correction(knots,state,&kg,&kh);
        for(int i=0;i<3;++i)gradient.a[i]+=kg.a[i];for(int i=0;i<9;++i)hessian.a[i]+=kh.a[i];
        auto update=cpu_newton(state,gradient,hessian);state=update.parameters;
        row(checkpoints,r,"delta_theta",state,update.accepted_step,update.halving_count);
      }
      broadcast(state,group);
      double seconds=MPI_Wtime()-start,max_seconds=0;
      // Timing is also leader-only; no non-leader enters federated collectives.
      if(group.is_leader)MPI_Reduce(&seconds,&max_seconds,1,MPI_DOUBLE,MPI_MAX,0,group.leaders_comm);
      double blocks[3]={mu_end-start,gamma_end-gamma_start,delta_end-delta_start},max_blocks[3]{};
      if(group.is_leader)MPI_Reduce(blocks,max_blocks,3,MPI_DOUBLE,MPI_MAX,0,group.leaders_comm);
      const std::uint64_t local_round_bytes=cpu_communication_bytes()-communication_begin;
      std::uint64_t global_round_bytes=0;
      MPI_Reduce(&local_round_bytes,&global_round_bytes,1,MPI_UINT64_T,MPI_SUM,0,MPI_COMM_WORLD);
      if(rank==0){timing<<r<<','<<std::setprecision(17)<<max_seconds;
        for(double t:max_blocks)timing<<','<<t;
        timing<<','<<global_round_bytes<<'\n';timing.flush();
        run_communication_bytes+=global_round_bytes;}
    }
    struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
    std::vector<long> rss(rank==0?o.workers*o.ranks:0);
    MPI_Gather(&usage.ru_maxrss,1,MPI_LONG,rss.data(),1,MPI_LONG,0,MPI_COMM_WORLD);
    if(rank==0) {
      std::ofstream memory(o.output/"peak_rss.csv");memory<<"rank,worker,worker_rank,peak_rss_kib\n";
      for(size_t i=0;i<rss.size();++i)memory<<i<<','<<i/o.ranks<<','<<i%o.ranks<<','<<rss[i]<<'\n';
      std::ofstream manifest(o.output/"manifest.txt");
      manifest<<"backend="<<o.backend<<"\nprecision=fp64\nworkers="<<o.workers<<"\nranks_per_worker="<<o.ranks
              <<"\ncpus_per_rank="<<o.threads<<"\nrounds="<<o.rounds<<"\nfixture="<<o.fixture
              <<"\nnu2="<<state.nu2<<"\ninit_delta="<<std::setprecision(17)<<initial_state.delta
              <<"\ninit_alpha="<<initial_state.alpha<<"\ninit_range="<<initial_state.range
              <<"\nmean_model="<<(o.zero_mean?"fixed_zero":"regression")
              <<"\naggregation=ordered_worker_leaders\n"
              <<"logical_communication_bytes="<<run_communication_bytes
              <<"\ncommunication_definition=sum_of_algorithm_payload_bytes_over_logical_sender_receiver_links;measurement_collectives_excluded\n"
#ifdef FEDSPATIAL_SCALAPACK
              <<"local_factorization=scalapack_column_block_cyclic_then_gather_leader\nworkspace=leader_dense_followers_distributed_operation_buffers\n"
#else
              <<"local_factorization=replicated_dense\n"
#endif
              ;
      if(!manifest)throw std::runtime_error("manifest write failed");
      std::cout<<"SYNCHRONOUS_TRAJECTORY_STATUS=PASS\n";
    }
    group.destroy();MPI_Finalize();return 0;
  } catch(const std::exception& e){std::cerr<<"MPI_RANK="<<rank<<" FAILURE="<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);return 2;}
}
