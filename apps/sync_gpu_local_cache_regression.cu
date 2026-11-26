#include "fedspatial/gpu_context.hpp"

#include <cmath>
#include <iostream>
#include <new>
#include <type_traits>

using namespace fedspatial;
static WorkerInput make_worker(double offset) {
  Matrix s(3,2),z(3,1),x(3,1);
  for(int i=0;i<3;++i){s(i,0)=offset+.2*i;s(i,1)=.15+.3*i;z(i,0)=offset-.4*i;x(i,0)=1.;}
  return {s,z,x};
}
static GlobalParameters parameters(){Matrix g(1,1),m(2,1),s(2,2);g(0,0)=.15;m(0,0)=.2;m(1,0)=-.1;s(0,0)=.7;s(0,1)=s(1,0)=.05;s(1,1)=.4;return {g,m,s,1.7,1.2,.8,3};}
int main(){try{Matrix k(2,2);k(0,0)=k(0,1)=0.;k(1,0)=k(1,1)=1.5;GpuContext cached(0),reference(0);using Slot=std::aligned_storage_t<sizeof(WorkerInput),alignof(WorkerInput)>;Slot slot;auto*a=new(&slot) WorkerInput(make_worker(0.));(void)compute_mu_sigma_local(cached,*a,k,parameters());a->~WorkerInput();auto*b=new(&slot) WorkerInput(make_worker(1.));const auto got=compute_mu_sigma_local(cached,*b,k,parameters());const WorkerInput independent=make_worker(1.);const auto want=compute_mu_sigma_local(reference,independent,k,parameters());double err=0.;for(size_t i=0;i<got.B.a.size();++i)err=std::max(err,std::abs(got.B.a[i]-want.B.a[i]));for(size_t i=0;i<got.M.a.size();++i)err=std::max(err,std::abs(got.M.a[i]-want.M.a[i]));b->~WorkerInput();std::cout<<"LOCAL_CACHE_REUSED_ADDRESS_MAX_ABS_ERROR="<<err<<'\n'<<"LOCAL_CACHE_REUSED_ADDRESS="<<(err==0.?"PASS":"FAIL")<<'\n';return err==0.?0:1;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
