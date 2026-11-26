#include "fedspatial/gpu_context.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace fedspatial; namespace fs=std::filesystem;
static Matrix load(const fs::path&p){Matrix x;if(!read_matrix(p.string(),&x))throw std::runtime_error("missing "+p.string());return x;}
int main(){fs::path f="sync_gpu/tests/fixtures/figure1_reference",o="results/figure1_cpp_gpu_faithful/parity";fs::create_directories(o);std::vector<WorkerInput>w;for(int id=0;id<10;id++){Matrix d=load(f/("worker_"+std::to_string(id)+".txt"));WorkerInput q{Matrix(100,2),Matrix(100,1),Matrix(100,5)};for(int i=0;i<100;i++){q.sites(i,0)=d(i,0);q.sites(i,1)=d(i,1);q.z(i,0)=d(i,2);for(int j=0;j<5;j++)q.x(i,j)=d(i,j+3);}w.push_back(q);}GlobalParameters p{Matrix(5,1),Matrix(),Matrix(),.25,1,.11};double c0[5]={-1,2,3,-2,1};for(int i=0;i<5;i++)p.gamma(i,0)=c0[i];GpuContext c(0);std::ofstream out(o/"ind_cpp_trajectory.csv");out<<"iteration,gamma0,gamma1,gamma2,gamma3,gamma4,delta,alpha,beta,accepted_step,halving_count\n";for(int it=0;it<=50;it++){out<<it<<','<<std::setprecision(17);for(int j=0;j<5;j++)out<<p.gamma(j,0)<<',';out<<p.delta<<','<<p.alpha<<','<<p.range<<",1,0\n";if(it<50)p=run_ind_synchronous_round(c,w,p).parameters;}return 0;}
