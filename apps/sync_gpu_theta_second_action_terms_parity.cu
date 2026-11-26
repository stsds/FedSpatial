#include "fedspatial/gpu_context.hpp"
#include "fedspatial/nworker_15k_figure1_config.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
using namespace fedspatial;
namespace {
double v(int i,int j){return std::sin(.071*(i+1)*(j+2))+.03*(i-j);}
WorkerInput worker(){Matrix s(37,2),z(37,1),x(37,1);for(int i=0;i<37;++i){s(i,0)=.013*i;s(i,1)=.021*(i%11);z(i,0)=v(i,3);x(i,0)=1.;}return{s,z,x};}
Matrix knots(){Matrix k(64,2);for(int j=0;j<64;++j){k(j,0)=.019*j+.004;k(j,1)=.017*(j%13)+.002;}return k;}
GlobalParameters parameters(){Matrix g(1,1),mu(64,1),sigma(64,64);g(0,0)=.1;for(int i=0;i<64;++i){mu(i,0)=.01*std::cos(.1*i);for(int j=0;j<64;++j)sigma(i,j)=(i==j?.9:0.)+.002*std::exp(-.03*std::abs(i-j));}return{g,mu,sigma,.7,1.15,.43};}
Matrix rhs(int columns){Matrix x(37,columns);for(int c=0;c<columns;++c)for(int r=0;r<37;++r)x.a[size_t(c)*37+r]=v(r,c);return x;}
void error(const Matrix&a,const Matrix&b,double*aa,double*rr){for(size_t k=0;k<a.a.size();++k){const double d=std::abs(a.a[k]-b.a[k]);*aa=std::max(*aa,d);*rr=std::max(*rr,d/std::max({1.,std::abs(a.a[k]),std::abs(b.a[k])}));}}
bool same(const std::array<Matrix,10>&a,const std::array<Matrix,10>&b){for(int i=0;i<10;++i)if(a[i].a.size()!=b[i].a.size()||std::memcmp(a[i].a.data(),b[i].a.data(),a[i].a.size()*sizeof(double)))return false;return true;}
}
int main(){try{setenv("CUDA_VISIBLE_DEVICES",NWorker15kFigure1Config::gv100_uuid_0,1);setenv("FEDSPATIAL_THETA_SECOND_TILE_CHECK_GUARDS","1",1);cudaDeviceProp p{};cudaGetDeviceProperties(&p,0);if(std::string(p.name).find("GV100")==std::string::npos)throw std::runtime_error("AF mask did not select GV100");const auto w=worker();const auto k=knots();const auto par=parameters();bool ok=true;for(const int width:{1,7,31,32,33,64,127,128,255,256,257,399,400,401,511,512})for(const auto ij:std::array<std::array<int,2>,3>{{{{0,0}},{{0,1}},{{1,1}}}}){const auto x=rhs(width);GpuContext c(0);const auto scalar=test_theta_a_second_action_terms(c,w,k,par,ij[0],ij[1],x,false);const auto tiled=test_theta_a_second_action_terms(c,w,k,par,ij[0],ij[1],x,true);const auto repeated=test_theta_a_second_action_terms(c,w,k,par,ij[0],ij[1],x,true);double ma=0.,mr=0.;for(int term=0;term<10;++term){double a=0.,r=0.;error(scalar[term],tiled[term],&a,&r);ma=std::max(ma,a);mr=std::max(mr,r);std::cout<<"SECOND_ACTION_TERM i="<<ij[0]<<" j="<<ij[1]<<" width="<<width<<" term="<<term<<" max_abs="<<a<<" max_rel="<<r<<'\n';}const bool pass=ma<=2e-10&&mr<=2e-10&&same(tiled,repeated);std::cout<<"SECOND_ACTION_COMPONENT i="<<ij[0]<<" j="<<ij[1]<<" width="<<width<<" max_abs="<<ma<<" max_rel="<<mr<<" byte_repeat="<<(same(tiled,repeated)?"PASS":"FAIL")<<" allocations="<<c.region().theta_second_tile_allocation_count<<" guard_checks="<<c.region().theta_second_tile_guard_checks<<" RESULT="<<(pass?"PASS":"FAIL")<<'\n';ok=ok&&pass;}std::cout<<"SECOND_ACTION_TERMS_PARITY="<<(ok?"PASS":"FAIL")<<'\n';return ok?0:1;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
