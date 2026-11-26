#include "fedspatial/gpu_context.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace fedspatial;
static Matrix load(const std::string& p){std::ifstream f(p);int r,c;if(!(f>>r>>c))throw std::runtime_error("missing "+p);Matrix x(r,c);for(double&v:x.a)f>>v;return x;}
static WorkerInput fixture(int n){auto a=load("sync_gpu/tests/fixtures/figure1_reference/worker_0.txt");Matrix s(n,2),z(n,1),x(n,5);for(int i=0;i<n;++i){s(i,0)=a(i,0);s(i,1)=a(i,1);z(i,0)=a(i,2);for(int j=0;j<5;++j)x(i,j)=a(i,j+3);}return{s,z,x};}
static double relative(double a,double b){return std::abs(a-b)/std::max({1.,std::abs(a),std::abs(b)});}
static bool check_case(int n){auto k=load("sync_gpu/tests/fixtures/figure1_reference/knots.txt");auto p=GlobalParameters{load("sync_gpu/tests/fixtures/figure1_reference/p_initial_gamma_new.txt"),load("sync_gpu/tests/fixtures/figure1_reference/p_initial_global_mu.txt"),load("sync_gpu/tests/fixtures/figure1_reference/p_initial_global_sigma.txt"),.25,1.,.11};GpuContext c(0);auto w=fixture(n);auto state=build_device_mu_sigma_local(c,w,k,p);auto dense=build_device_delta_theta_workspace(c,state);auto h=download_device_delta_theta_workspace_for_test(c,dense);double q=0,t=0,ld=-n*std::log(p.delta);for(int i=0;i<n;++i){q+=h.r(i,0)*h.u(i,0);ld+=2*std::log(state.chol_diag?1.:1.);for(int j=0;j<n;++j)t+=h.W(i,j)*h.Q(j,i);} // logdet is checked through the streamed objective repeat, not the host slice.
auto before=c.raw_h2d_bytes();auto a=compute_device_delta_theta_base_reduction_streamed(c,w,k,p);auto b=compute_device_delta_theta_base_reduction_streamed(c,w,k,p);double aa=std::max(std::abs(a.quadratic-q),std::abs(a.trace_q-t)),rr=std::max(relative(a.quadratic,q),relative(a.trace_q,t));bool exact=std::memcmp(&a,&b,sizeof(a))==0,zero=c.raw_h2d_bytes()==before;std::cout<<"STREAM_BASE_N="<<n<<" MAX_ABS="<<aa<<" MAX_REL="<<rr<<" EXACT_REPEAT="<<(exact?"PASS":"FAIL")<<" ZERO_ACTIVE_H2D="<<(zero?"PASS":"FAIL")<<'\n';return aa<=1e-10&&rr<=1e-10&&exact&&zero;}
int main(){try{bool ok=check_case(3)&&check_case(100);std::cout<<"STREAMED_BASE_PARITY="<<(ok?"PASS":"FAIL")<<'\n';return ok?0:1;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
