#include "fedspatial/gpu_context.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fedspatial;

namespace {

struct Expected { int round{}; std::string stage; std::vector<double> value; };
struct Error { double abs{}, rel{}; void add(double x,double y){abs=std::max(abs,std::abs(x-y));rel=std::max(rel,std::abs(x-y)/std::max(1.,std::abs(y)));} };
struct Errors { Error gamma,mu,sigma,delta,alpha,range; };

Matrix load(const std::string& path) {
  Matrix value;
  if (!read_matrix(path, &value)) throw std::runtime_error("missing " + path);
  return value;
}

WorkerInput worker(int rank) {
  Matrix data = load("sync_gpu/tests/fixtures/sync_gpu/N_WORKER_" + std::to_string(rank) + ".txt");
  WorkerInput result{Matrix(data.rows,2), Matrix(data.rows,1), Matrix(data.rows,data.cols-3)};
  for (int i=0;i<data.rows;++i) {
    result.sites(i,0)=data(i,0); result.sites(i,1)=data(i,1); result.z(i,0)=data(i,2);
    for (int j=0;j<result.x.cols;++j) result.x(i,j)=data(i,j+3);
  }
  return result;
}

Matrix knots() { Matrix k(2,2); k(0,0)=k(0,1)=0.; k(1,0)=k(1,1)=1.5; return k; }
GlobalParameters initial() {
  Matrix g(1,1),mu(2,1),sigma(2,2); g(0,0)=.15; mu(0,0)=.2; mu(1,0)=-.1;
  sigma(0,0)=.7; sigma(0,1)=sigma(1,0)=.05; sigma(1,1)=.4;
  return {g,mu,sigma,1.7,1.2,.8,3};
}

std::vector<Expected> expected() {
  std::ifstream in("sync_gpu/tests/fixtures/sync_gpu/N_WORKER_2_P_CHECKPOINTS.csv");
  if (!in) throw std::runtime_error("missing two-worker Python checkpoints");
  std::string line,field; std::getline(in,line); std::vector<Expected> rows;
  while(std::getline(in,line)) { std::stringstream s(line); Expected row;
    std::getline(s,field,','); row.round=std::stoi(field); std::getline(s,row.stage,',');
    while(std::getline(s,field,',')) row.value.push_back(std::stod(field));
    if(row.value.size()!=12) throw std::runtime_error("bad checkpoint row"); rows.push_back(std::move(row));
  }
  return rows;
}

std::vector<double> pack(const GlobalParameters& p) {
  std::vector<double> v; v.reserve(p.gamma.a.size()+p.mu.a.size()+p.sigma.a.size()+4);
  v.insert(v.end(),p.gamma.a.begin(),p.gamma.a.end()); v.insert(v.end(),p.mu.a.begin(),p.mu.a.end());
  v.insert(v.end(),p.sigma.a.begin(),p.sigma.a.end()); v.push_back(p.delta); v.push_back(p.alpha);
  v.push_back(p.range); v.push_back(double(p.nu2)); return v;
}
void unpack(const std::vector<double>& v, GlobalParameters* p) {
  size_t at=0; for(double& x:p->gamma.a)x=v.at(at++); for(double& x:p->mu.a)x=v.at(at++);
  for(double& x:p->sigma.a)x=v.at(at++); p->delta=v.at(at++);p->alpha=v.at(at++);p->range=v.at(at++);p->nu2=int(v.at(at++));
}
void broadcast(GlobalParameters* p, int root, MPI_Comm comm) {
  auto values=pack(*p); MPI_Bcast(values.data(),int(values.size()),MPI_DOUBLE,root,comm); unpack(values,p);
}

void add(Matrix* a,const Matrix& b){if(a->a.size()!=b.a.size())throw std::runtime_error("matrix shape mismatch");for(size_t i=0;i<a->a.size();++i)a->a[i]+=b.a[i];}

struct Eigen3 { std::array<double,3> values{}; Matrix vectors{3,3}; };
Eigen3 symmetric_eigen3(Matrix a) {
  Eigen3 out; for(int i=0;i<3;++i)out.vectors(i,i)=1.;
  for(int it=0;it<64;++it){int p=0,q=1;double mx=std::abs(a(0,1));for(int i=0;i<3;++i)for(int j=i+1;j<3;++j)if(std::abs(a(i,j))>mx){mx=std::abs(a(i,j));p=i;q=j;}if(mx<1.e-15)break;
    const double angle=.5*std::atan2(2.*a(p,q),a(q,q)-a(p,p)),c=std::cos(angle),s=std::sin(angle);
    for(int k=0;k<3;++k){double pk=a(p,k),qk=a(q,k);a(p,k)=c*pk-s*qk;a(q,k)=s*pk+c*qk;}
    for(int k=0;k<3;++k){double kp=a(k,p),kq=a(k,q);a(k,p)=c*kp-s*kq;a(k,q)=s*kp+c*kq;double vp=out.vectors(k,p),vq=out.vectors(k,q);out.vectors(k,p)=c*vp-s*vq;out.vectors(k,q)=s*vp+c*vq;}}
  std::array<int,3> order{0,1,2};std::sort(order.begin(),order.end(),[&](int i,int j){return a(i,i)<a(j,j);});Matrix v=out.vectors;
  for(int k=0;k<3;++k){out.values[k]=a(order[k],order[k]);for(int i=0;i<3;++i)out.vectors(i,k)=v(i,order[k]);}return out;
}

void compare(const GlobalParameters& p,const Expected& e,Errors* x){x->gamma.add(p.gamma(0,0),e.value[0]);x->mu.add(p.mu(0,0),e.value[1]);x->mu.add(p.mu(1,0),e.value[2]);x->sigma.add(p.sigma(0,0),e.value[3]);x->sigma.add(p.sigma(0,1),e.value[4]);x->sigma.add(p.sigma(1,0),e.value[5]);x->sigma.add(p.sigma(1,1),e.value[6]);x->delta.add(p.delta,e.value[7]);x->alpha.add(p.alpha,e.value[8]);x->range.add(p.range,e.value[9]);}

} // namespace

int main(int argc,char** argv) {
  MPI_Init(&argc,&argv); int rank=0,size=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
  try {
    if(size!=2)throw std::runtime_error("parity test requires exactly two MPI ranks");
    MPI_Comm local_comm; MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local_comm);
    int local_rank=0;MPI_Comm_rank(local_comm,&local_rank);MPI_Comm_free(&local_comm);
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||local_rank>=devices)throw std::runtime_error("one visible GPU is required per local MPI rank");
    GpuContext context(local_rank); const auto input=worker(rank); const auto knot=knots(); auto state=initial();
    const auto reference=rank==0?expected():std::vector<Expected>{}; Errors errors{}; int checkpoint=0; bool step_ok=true;
    for(int round=0;round<2;++round){
      const auto local_mu=compute_mu_sigma_local(context,input,knot,state).summary; const int m=knot.rows;
      std::vector<double> send_mu;send_mu.insert(send_mu.end(),local_mu.mu.a.begin(),local_mu.mu.a.end());send_mu.insert(send_mu.end(),local_mu.sigma.a.begin(),local_mu.sigma.a.end());
      std::vector<double> gathered_mu(rank==0?send_mu.size()*size:0);MPI_Gather(send_mu.data(),int(send_mu.size()),MPI_DOUBLE,gathered_mu.data(),int(send_mu.size()),MPI_DOUBLE,0,MPI_COMM_WORLD);
      if(rank==0){Matrix mean_mu(m,1),mean_sigma(m,m);for(int r=0;r<size;++r){size_t off=size_t(r)*send_mu.size();for(int i=0;i<m;++i)mean_mu.a[i]+=gathered_mu[off+i];for(int i=0;i<m*m;++i)mean_sigma.a[i]+=gathered_mu[off+m+i];}for(double&v:mean_mu.a)v/=size;for(double&v:mean_sigma.a)v/=size;
        Matrix k=matern_with_derivatives(context,knot,knot,state.nu2,state.alpha,state.range).value;Matrix precision=inverse_spd(k);for(size_t i=0;i<precision.a.size();++i)precision.a[i]+=size*state.delta*mean_sigma.a[i];state.sigma=inverse_spd(precision);state.mu=matmul(state.sigma,mean_mu);for(double&v:state.mu.a)v*=-size*state.delta;compare(state,reference.at(checkpoint++),&errors);}
      broadcast(&state,0,MPI_COMM_WORLD);

      const auto local_gamma=compute_gamma_local(context,input,knot,state);const int q=state.gamma.rows;
      std::vector<double> send_gamma;send_gamma.insert(send_gamma.end(),local_gamma.gamma.a.begin(),local_gamma.gamma.a.end());send_gamma.insert(send_gamma.end(),local_gamma.xx.a.begin(),local_gamma.xx.a.end());
      std::vector<double> gathered_gamma(rank==0?send_gamma.size()*size:0);MPI_Gather(send_gamma.data(),int(send_gamma.size()),MPI_DOUBLE,gathered_gamma.data(),int(send_gamma.size()),MPI_DOUBLE,0,MPI_COMM_WORLD);
      if(rank==0){Matrix rhs(q,1),xx(q,q);for(int r=0;r<size;++r){size_t off=size_t(r)*send_gamma.size();for(int i=0;i<q;++i)rhs.a[i]+=gathered_gamma[off+i];for(int i=0;i<q*q;++i)xx.a[i]+=gathered_gamma[off+q+i];}for(double&v:rhs.a)v/=size;for(double&v:xx.a)v/=size;state.gamma=solve_spd(xx,rhs);compare(state,reference.at(checkpoint++),&errors);}
      broadcast(&state,0,MPI_COMM_WORLD);

      const auto local_theta=compute_delta_theta_local(context,input,knot,state);std::array<double,13> send_theta{};send_theta[0]=local_theta.objective;for(int i=0;i<3;++i)send_theta[1+i]=local_theta.gradient[i];for(int i=0;i<9;++i)send_theta[4+i]=local_theta.hessian[i];
      std::vector<double> gathered_theta(rank==0?13*size:0);MPI_Gather(send_theta.data(),13,MPI_DOUBLE,gathered_theta.data(),13,MPI_DOUBLE,0,MPI_COMM_WORLD);
      double accepted_step=1.;int halvings=0;
      if(rank==0){Matrix gradient(3,1),hessian(3,3);for(int r=0;r<size;++r){const double* v=gathered_theta.data()+13*r;for(int i=0;i<3;++i)gradient(i,0)+=v[1+i];for(int i=0;i<9;++i)hessian.a[i]+=v[4+i];}
        Matrix kg(3,1),kh(3,3);compute_legacy_global_k_correction_for_test(context,knot,state,&kg,&kh);add(&gradient,kg);add(&hessian,kh);
        const auto eigen=symmetric_eigen3(hessian);Matrix modified(3,3);for(int k=0;k<3;++k){double value=std::max(std::abs(eigen.values[k]),.01);for(int i=0;i<3;++i)for(int j=0;j<3;++j)modified(i,j)+=eigen.vectors(i,k)*value*eigen.vectors(j,k);}Matrix direction=solve_spd(modified,gradient);
        double d,a,b;do{d=state.delta-accepted_step*direction(0,0);a=state.alpha-accepted_step*direction(1,0);b=state.range-accepted_step*direction(2,0);if(d<=0.||a<=0.||b<=0.){accepted_step*=.5;++halvings;}else break;}while(true);state.delta=d;state.alpha=a;state.range=b;const auto& e=reference.at(checkpoint++);compare(state,e,&errors);step_ok=step_ok&&std::abs(accepted_step-e.value[10])<=1.e-15&&halvings==int(e.value[11]);}
      broadcast(&state,0,MPI_COMM_WORLD);
    }
    int result=0;if(rank==0){double maximum=std::max({errors.gamma.abs,errors.mu.abs,errors.sigma.abs,errors.delta.abs,errors.alpha.abs,errors.range.abs});auto print=[](const char*n,const Error&e){std::cout<<n<<"_MAX_ABS="<<e.abs<<" MAX_REL="<<e.rel<<'\n';};
      std::cout<<std::setprecision(17)<<"MPI_RANKS="<<size<<"\nMPI_ONE_RANK_PER_GPU=PASS\n";print("MPI_GAMMA",errors.gamma);print("MPI_MU",errors.mu);print("MPI_SIGMA",errors.sigma);print("MPI_DELTA",errors.delta);print("MPI_ALPHA",errors.alpha);print("MPI_RANGE",errors.range);std::cout<<"MPI_NEWTON_STEP_PARITY="<<(step_ok?"PASS":"FAIL")<<'\n'<<"MPI_P_PARITY="<<(maximum<=1.e-10&&step_ok?"PASS":"FAIL")<<'\n';result=maximum<=1.e-10&&step_ok?0:1;}
    MPI_Bcast(&result,1,MPI_INT,0,MPI_COMM_WORLD);MPI_Finalize();return result;
  }catch(const std::exception& e){std::cerr<<"MPI_RANK="<<rank<<" FAILURE="<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,2);return 2;}
}
