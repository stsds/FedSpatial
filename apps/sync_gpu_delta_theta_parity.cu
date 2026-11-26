#include "fedspatial/gpu_context.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace fedspatial;
static bool load(const std::string& n,Matrix* x){return read_matrix("sync_gpu/tests/fixtures/sync_gpu/"+n,x);}
int main(){constexpr int n=3,m=2;Matrix k(m,2),s(n,2),z(n,1),x(n,1),g(1,1),mu(m,1),sig(m,m);double hs[]={0,0,1,.2,.3,1.1},hk[]={0,0,1.5,1.5},hz[]={1.2,-.4,.7},hx[]={1,1,1};std::copy(hs,hs+6,s.a.begin());std::copy(hk,hk+4,k.a.begin());std::copy(hz,hz+3,z.a.begin());std::copy(hx,hx+3,x.a.begin());g(0,0)=.15;mu(0,0)=.2;mu(1,0)=-.1;sig(0,0)=.7;sig(0,1)=sig(1,0)=.05;sig(1,1)=.4;WorkerInput w{s,z,x};GpuContext context(0);double worst=0,worst_rel=0;for(int i=0;i<7;i++){Matrix v,ref;if(!load("DELTA_THETA_POINT"+std::to_string(i)+"_PARAM.txt",&v)||!load("DELTA_THETA_POINT"+std::to_string(i)+"_OBJECTIVE.txt",&ref))return 2;GlobalParameters p{g,mu,sig,v(0,0),v(1,0),v(2,0)};const double got=compute_delta_theta_local(context,w,k,p).objective;double e=std::abs(got-ref(0,0));worst=std::max(worst,e);worst_rel=std::max(worst_rel,e/std::max(1e-300,std::abs(ref(0,0))));std::cout<<"DELTA_THETA_OBJECTIVE_POINT"<<i<<"_ERROR="<<e<<"\n";}bool ok=worst<=1e-10;std::cout<<"DELTA_THETA_OBJECTIVE_MAX_ABS_ERROR="<<worst<<"\nDELTA_THETA_OBJECTIVE_MAX_REL_ERROR="<<worst_rel<<"\nDELTA_THETA_OBJECTIVE_PARITY="<<(ok?"PASS":"FAIL")<<"\n";return ok?0:1;}
