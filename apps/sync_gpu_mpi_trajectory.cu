#include "fedspatial/figure1_fixture.hpp"
#include "fedspatial/gpu_context.hpp"
#include <mpi.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fedspatial;
namespace fs=std::filesystem;
namespace {
struct Opt { fs::path fixture,output; int workers{},rounds{50},gpus_per_node{},gpus_per_worker{1},nu2{3}; double beta{.11},init_delta{.25},init_alpha{1.},init_range{-1.}; std::string precision{"fp64"}; };
Opt options(int argc,char**argv){Opt o;for(int i=1;i<argc;++i){std::string f=argv[i];if(i+1>=argc)throw std::runtime_error("missing value for "+f);std::string v=argv[++i];if(f=="--fixture-dir")o.fixture=v;else if(f=="--output-dir")o.output=v;else if(f=="--workers")o.workers=std::stoi(v);else if(f=="--rounds")o.rounds=std::stoi(v);else if(f=="--gpus-per-node")o.gpus_per_node=std::stoi(v);else if(f=="--gpus-per-worker")o.gpus_per_worker=std::stoi(v);else if(f=="--nu2")o.nu2=std::stoi(v);else if(f=="--range")o.beta=std::stod(v);else if(f=="--init-delta")o.init_delta=std::stod(v);else if(f=="--init-alpha")o.init_alpha=std::stod(v);else if(f=="--init-range")o.init_range=std::stod(v);else if(f=="--precision")o.precision=v;else throw std::runtime_error("unknown option "+f);}if(o.fixture.empty()||o.output.empty()||o.workers<1||o.rounds<1)throw std::runtime_error("require fixture/output/workers and positive rounds");if(o.init_range<0)o.init_range=o.beta;if(!(o.init_delta>0&&o.init_alpha>0&&o.init_range>0))throw std::runtime_error("initial covariance parameters must be positive");if(o.gpus_per_worker!=1)throw std::runtime_error("current exact algorithm requires --gpus-per-worker 1; multi-GPU local covariance factorization is not implemented");if(o.precision!="fp64"&&o.precision!="fp32"&&o.precision!="fp16_fp32")throw std::runtime_error("precision must be fp64, fp32, or fp16_fp32");return o;}
GlobalParameters initial(int q,int m,int nu2,double delta,double alpha,double range){Matrix g(q,1),mu(m,1),sigma(m,m);const double a[5]={-1,2,3,-2,1};if(q!=5)throw std::runtime_error("Figure-7 driver currently requires five covariates");for(int i=0;i<q;++i)g(i,0)=a[i];return {g,mu,sigma,delta,alpha,range,nu2};}
std::vector<double> pack(const GlobalParameters&p){std::vector<double>v;v.insert(v.end(),p.gamma.a.begin(),p.gamma.a.end());v.insert(v.end(),p.mu.a.begin(),p.mu.a.end());v.insert(v.end(),p.sigma.a.begin(),p.sigma.a.end());v.push_back(p.delta);v.push_back(p.alpha);v.push_back(p.range);return v;}
void unpack(const std::vector<double>&v,GlobalParameters*p){size_t k=0;for(double&x:p->gamma.a)x=v.at(k++);for(double&x:p->mu.a)x=v.at(k++);for(double&x:p->sigma.a)x=v.at(k++);p->delta=v.at(k++);p->alpha=v.at(k++);p->range=v.at(k++);}
void bcast(GlobalParameters*p){auto v=pack(*p);MPI_Bcast(v.data(),int(v.size()),MPI_DOUBLE,0,MPI_COMM_WORLD);unpack(v,p);}
struct Eig {std::array<double,3>d{};Matrix v{3,3};};
Eig eig3(Matrix a){Eig e;for(int i=0;i<3;++i)e.v(i,i)=1.;for(int z=0;z<64;++z){int p=0,q=1;double x=std::abs(a(0,1));for(int i=0;i<3;++i)for(int j=i+1;j<3;++j)if(std::abs(a(i,j))>x){x=std::abs(a(i,j));p=i;q=j;}if(x<1e-15)break;double t=.5*std::atan2(2*a(p,q),a(q,q)-a(p,p)),c=std::cos(t),s=std::sin(t);for(int k=0;k<3;++k){double u=a(p,k),w=a(q,k);a(p,k)=c*u-s*w;a(q,k)=s*u+c*w;}for(int k=0;k<3;++k){double u=a(k,p),w=a(k,q);a(k,p)=c*u-s*w;a(k,q)=s*u+c*w;u=e.v(k,p);w=e.v(k,q);e.v(k,p)=c*u-s*w;e.v(k,q)=s*u+c*w;}}std::array<int,3>o{0,1,2};std::sort(o.begin(),o.end(),[&](int i,int j){return a(i,i)<a(j,j);});Matrix v=e.v;for(int k=0;k<3;++k){e.d[k]=a(o[k],o[k]);for(int i=0;i<3;++i)e.v(i,k)=v(i,o[k]);}return e;}
double max_time(double local){double v=0;MPI_Reduce(&local,&v,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);return v;}
void row(std::ofstream&f,int it,double wall,double objective,const GlobalParameters&p,double step,int halves){f<<it<<','<<std::setprecision(17)<<wall<<','<<objective;for(double x:p.gamma.a)f<<','<<x;f<<','<<p.delta<<','<<p.alpha<<','<<p.range<<','<<step<<','<<halves<<'\n';f.flush();}
MuSigmaLocalSummary compact_mu_sigma_summary(GpuContext& context,
                                             const DeviceMuSigmaLocalState& state) {
  MuSigmaLocalSummary out{Matrix(state.m,1),Matrix(state.m,state.m)};
  auto copy=[&](void* destination,const void* source,size_t bytes) {
    const auto status=cudaMemcpyAsync(destination,source,bytes,cudaMemcpyDeviceToHost,
                                      context.stream());
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
  };
  copy(out.mu.a.data(),state.bt_minv_e,size_t(state.m)*sizeof(double));
  copy(out.sigma.a.data(),state.bt_minv_B,size_t(state.m)*state.m*sizeof(double));
  const auto sync=cudaStreamSynchronize(context.stream());
  if(sync!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(sync));
  context.record_model_d2h_bytes(
      (size_t(state.m)+size_t(state.m)*state.m)*sizeof(double));
  for(double& value:out.mu.a)value=-value/state.delta;
  for(double& value:out.sigma.a)value/=state.delta;
  return out;
}
}
int main(int argc,char**argv){MPI_Init(&argc,&argv);int rank=0,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);try{const auto o=options(argc,argv);if(size!=o.workers)throw std::runtime_error("MPI size must equal --workers");MPI_Comm local;MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local);int lr=0,ls=0;MPI_Comm_rank(local,&lr);MPI_Comm_size(local,&ls);MPI_Comm_free(&local);int nd=0;cudaGetDeviceCount(&nd);if(lr>=nd)throw std::runtime_error("local MPI rank has no GPU");if(o.gpus_per_node&&ls!=o.gpus_per_node)throw std::runtime_error("local ranks do not match --gpus-per-node");cudaSetDevice(lr);char host[MPI_MAX_PROCESSOR_NAME]{};int hn=0;MPI_Get_processor_name(host,&hn);std::cout<<"MPI_MAP rank="<<rank<<" local_rank="<<lr<<" node="<<host<<" device="<<lr<<std::endl;
Figure1Fixture fixture;if(!load_figure1_python_fixture(o.fixture,&fixture))throw std::runtime_error("fixture load failed");auto workers=partition_figure1_fixture(fixture,size);const auto input=workers.at(rank);auto state=initial(fixture.x.cols,fixture.knots.rows,o.nu2,o.init_delta,o.init_alpha,o.init_range);GpuContext context(lr);PrecisionPolicy policy;if(o.precision=="fp32")policy.mode=ArithmeticPrecision::fp32;else if(o.precision=="fp16_fp32")policy.mode=ArithmeticPrecision::mixed_fp16_fp32;context.set_precision_policy(policy);std::ofstream trajectory,timing;if(rank==0){fs::create_directories(o.output);trajectory.open(o.output/"mpi_trajectory.csv");trajectory<<"iteration,wall_seconds,local_objective_sum,gamma0,gamma1,gamma2,gamma3,gamma4,delta,alpha,range,accepted_step,halving_count\n";timing.open(o.output/"mpi_timing.csv");timing<<"iteration,mu_sigma_seconds,gamma_seconds,delta_theta_seconds,total_seconds\n";row(trajectory,0,0,0,state,0,0);}MPI_Barrier(MPI_COMM_WORLD);double start=MPI_Wtime();
for(int it=0;it<o.rounds;++it){double round0=MPI_Wtime(),t0=round0;auto lm=compact_mu_sigma_summary(context,build_device_mu_sigma_local(context,input,fixture.knots,state));std::vector<double>s; s.insert(s.end(),lm.mu.a.begin(),lm.mu.a.end());s.insert(s.end(),lm.sigma.a.begin(),lm.sigma.a.end());std::vector<double>g(rank==0?s.size()*size:0);MPI_Gather(s.data(),s.size(),MPI_DOUBLE,g.data(),s.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);if(rank==0){int m=fixture.knots.rows;Matrix a(m,1),b(m,m);for(int r=0;r<size;++r){size_t z=r*s.size();for(int i=0;i<m;++i)a.a[i]+=g[z+i];for(int i=0;i<m*m;++i)b.a[i]+=g[z+m+i];}for(double&x:a.a)x/=size;for(double&x:b.a)x/=size;Matrix k=matern_with_derivatives(context,fixture.knots,fixture.knots,state.nu2,state.alpha,state.range).value,pr=inverse_spd(k);for(size_t i=0;i<pr.a.size();++i)pr.a[i]+=size*state.delta*b.a[i];state.sigma=inverse_spd(pr);state.mu=matmul(state.sigma,a);for(double&x:state.mu.a)x*=-size*state.delta;}bcast(&state);double tm=max_time(MPI_Wtime()-t0);
t0=MPI_Wtime();auto lg=compute_gamma_local(context,input,fixture.knots,state);s.clear();s.insert(s.end(),lg.gamma.a.begin(),lg.gamma.a.end());s.insert(s.end(),lg.xx.a.begin(),lg.xx.a.end());g.assign(rank==0?s.size()*size:0,0);MPI_Gather(s.data(),s.size(),MPI_DOUBLE,g.data(),s.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);if(rank==0){int q=state.gamma.rows;Matrix a(q,1),b(q,q);for(int r=0;r<size;++r){size_t z=r*s.size();for(int i=0;i<q;++i)a.a[i]+=g[z+i];for(int i=0;i<q*q;++i)b.a[i]+=g[z+q+i];}for(double&x:a.a)x/=size;for(double&x:b.a)x/=size;state.gamma=solve_spd(b,a);}bcast(&state);double tg=max_time(MPI_Wtime()-t0);
t0=MPI_Wtime();auto lt=compute_delta_theta_local(context,input,fixture.knots,state);std::array<double,13>s13{};s13[0]=lt.objective;for(int i=0;i<3;++i)s13[1+i]=lt.gradient[i];for(int i=0;i<9;++i)s13[4+i]=lt.hessian[i];std::vector<double>g13(rank==0?13*size:0);MPI_Gather(s13.data(),13,MPI_DOUBLE,g13.data(),13,MPI_DOUBLE,0,MPI_COMM_WORLD);double objective=0,step=1;int halves=0;if(rank==0){Matrix gr(3,1),h(3,3);for(int r=0;r<size;++r){const double*v=g13.data()+13*r;objective+=v[0];for(int i=0;i<3;++i)gr(i,0)+=v[1+i];for(int i=0;i<9;++i)h.a[i]+=v[4+i];}Matrix kg(3,1),kh(3,3);compute_legacy_global_k_correction_for_test(context,fixture.knots,state,&kg,&kh);for(int i=0;i<3;++i)gr.a[i]+=kg.a[i];for(int i=0;i<9;++i)h.a[i]+=kh.a[i];auto e=eig3(h);Matrix mh(3,3);for(int k=0;k<3;++k){double d=std::max(std::abs(e.d[k]),.01);for(int i=0;i<3;++i)for(int j=0;j<3;++j)mh(i,j)+=e.v(i,k)*d*e.v(j,k);}Matrix d=solve_spd(mh,gr);double x,y,z;do{x=state.delta-step*d(0,0);y=state.alpha-step*d(1,0);z=state.range-step*d(2,0);if(x<=0||y<=0||z<=0){step*=.5;++halves;}else break;}while(true);state.delta=x;state.alpha=y;state.range=z;}bcast(&state);MPI_Bcast(&objective,1,MPI_DOUBLE,0,MPI_COMM_WORLD);MPI_Bcast(&step,1,MPI_DOUBLE,0,MPI_COMM_WORLD);MPI_Bcast(&halves,1,MPI_INT,0,MPI_COMM_WORLD);double tt=max_time(MPI_Wtime()-t0),total=max_time(MPI_Wtime()-round0);if(rank==0){row(trajectory,it+1,MPI_Wtime()-start,objective,state,step,halves);timing<<it<<','<<tm<<','<<tg<<','<<tt<<','<<total<<'\n';timing.flush();}}
long long local_precision[2]{context.region().precision_gemm_calls,context.region().precision_promotions};
long long global_precision[2]{};
MPI_Reduce(local_precision,global_precision,2,MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
if(rank==0){std::ofstream m(o.output/"mpi_manifest.txt");m<<"workers="<<size<<"\ngpus_per_worker=1\ngpus_per_node="<<o.gpus_per_node<<"\nrounds="<<o.rounds<<"\nprecision="<<o.precision<<"\nreference_precision=fp64_gpu\ninit_delta="<<std::setprecision(17)<<o.init_delta<<"\ninit_alpha="<<o.init_alpha<<"\ninit_range="<<o.init_range<<"\nprecision_gemm_calls="<<global_precision[0]<<"\nprecision_promotions="<<global_precision[1]<<"\nfixture="<<o.fixture<<"\n";std::cout<<"PRECISION_STATS mode="<<o.precision<<" gemm_calls="<<global_precision[0]<<" promotions="<<global_precision[1]<<"\nMPI_TRAJECTORY_STATUS=PASS\n";}MPI_Finalize();return 0;}catch(const std::exception&e){std::cerr<<"MPI_RANK="<<rank<<" FAILURE="<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);return 2;}}
