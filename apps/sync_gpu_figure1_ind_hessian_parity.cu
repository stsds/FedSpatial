#include "fedspatial/gpu_context.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace fedspatial; namespace fs=std::filesystem;
static Matrix load(const fs::path&p){Matrix x;if(!read_matrix(p.string(),&x))throw std::runtime_error("missing "+p.string());return x;}
static WorkerInput wload(const fs::path&f,int id){Matrix d=load(f/("worker_"+std::to_string(id)+".txt"));WorkerInput w{Matrix(100,2),Matrix(100,1),Matrix(100,5)};for(int i=0;i<100;i++){w.sites(i,0)=d(i,0);w.sites(i,1)=d(i,1);w.z(i,0)=d(i,2);for(int j=0;j<5;j++)w.x(i,j)=d(i,j+3);}return w;}
int main(){fs::path f="sync_gpu/tests/fixtures/figure1_reference";GpuContext c(0);GlobalParameters p{load(f/"ind_initial_gamma_new.txt"),Matrix(),Matrix(),.25,1.,.11};double gerr=0,herr=0,oerr=0;std::ofstream csv("results/figure1_cpp_gpu_faithful/parity/ind_local_objective_parity.csv");csv<<"worker,python_value,cpp_value,abs_error\n";for(int id=0;id<10;id++){auto wi=wload(f,id);auto r=compute_ind_delta_theta_local(c,wi,p);Matrix po=load(f/("ind_initial_worker_"+std::to_string(id)+"_objective.txt"));double e=std::abs(po(0,0)-r.objective);oerr=std::max(oerr,e);csv<<id<<","<<po(0,0)<<","<<r.objective<<","<<e<<"\n";Matrix pg=load(f/("ind_initial_worker_"+std::to_string(id)+"_gradient.txt")),ph=load(f/("ind_initial_worker_"+std::to_string(id)+"_hessian.txt"));for(int i=0;i<3;i++){gerr=std::max(gerr,std::abs(pg(i,0)-r.gradient[i]));for(int j=0;j<3;j++)herr=std::max(herr,std::abs(ph(i,j)-r.hessian[3*i+j]));}}std::cout<<"IND_OBJECTIVE_MAX_ERROR="<<oerr<<"\nIND_GRADIENT_MAX_ERROR="<<gerr<<"\nIND_HESSIAN_MAX_ERROR="<<herr<<"\nIND_HESSIAN_PARITY="<<(std::max({oerr,gerr,herr})<1e-8?"PASS":"FAIL")<<"\n";return std::max({oerr,gerr,herr})<1e-8?0:1;}
