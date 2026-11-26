#include "fedspatial/gpu_context.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace fedspatial;namespace fs=std::filesystem;
static Matrix L(const fs::path&p){Matrix x;if(!read_matrix(p.string(),&x))throw std::runtime_error("missing");return x;}static WorkerInput W(const fs::path&f){Matrix d=L(f/"worker_0.txt");WorkerInput w{Matrix(100,2),Matrix(100,1),Matrix(100,5)};for(int i=0;i<100;i++){w.sites(i,0)=d(i,0);w.sites(i,1)=d(i,1);w.z(i,0)=d(i,2);for(int j=0;j<5;j++)w.x(i,j)=d(i,j+3);}return w;}
int main(){fs::path f="sync_gpu/tests/fixtures/figure1_reference",o="results/figure1_cpp_gpu_faithful/parity";GpuContext c(0);GlobalParameters p{L(f/"ind_initial_gamma_new.txt"),Matrix(),Matrix(),.25,1,.11};auto w=W(f);auto a=compute_ind_delta_theta_local(c,w,p);std::ofstream out(o/"ind_finite_difference_check.csv");out<<"epsilon,quantity,i,j,analytic,finite_difference,abs_error\n";for(double ep:{1e-5,1e-6}){for(int i=0;i<3;i++){double t=i==0?p.delta:(i==1?p.alpha:p.range),h=ep*std::max(1.,std::abs(t));auto pp=p,pm=p;if(i==0){pp.delta=t+h;pm.delta=t-h;}if(i==1){pp.alpha=t+h;pm.alpha=t-h;}if(i==2){pp.range=t+h;pm.range=t-h;}auto rp=compute_ind_delta_theta_local(c,w,pp),rm=compute_ind_delta_theta_local(c,w,pm);double fd=(rp.objective-rm.objective)/(2*h);out<<ep<<",gradient,"<<i<<",0,"<<a.gradient[i]<<","<<fd<<","<<std::abs(fd-a.gradient[i])<<"\n";for(int j=0;j<3;j++){double fh=(rp.gradient[j]-rm.gradient[j])/(2*h);out<<ep<<",hessian,"<<j<<","<<i<<","<<a.hessian[3*j+i]<<","<<fh<<","<<std::abs(fh-a.hessian[3*j+i])<<"\n";}}}return 0;}
