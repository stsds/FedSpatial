#include "fedspatial/gpu_context.hpp"
#include "fedspatial/nworker_15k_figure1_config.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace fedspatial;

namespace {
constexpr double kAbsTolerance=2e-10;
constexpr double kRelTolerance=2e-10;

double value(int i,int j) { return std::sin(.071*(i+1)*(j+2))+.03*(i-j); }

WorkerInput make_worker(int n) {
  Matrix sites(n,2),z(n,1),x(n,1);
  for(int i=0;i<n;++i) { sites(i,0)=.013*i; sites(i,1)=.021*(i%11); z(i,0)=value(i,3); x(i,0)=1.; }
  return {sites,z,x};
}
Matrix make_knots(int m) {
  Matrix k(m,2); for(int j=0;j<m;++j) { k(j,0)=.019*j+.004; k(j,1)=.017*(j%13)+.002; } return k;
}
GlobalParameters make_parameters(int m) {
  Matrix gamma(1,1),mu(m,1),sigma(m,m); gamma(0,0)=.1;
  for(int i=0;i<m;++i) { mu(i,0)=.01*std::cos(.1*i); for(int j=0;j<m;++j) sigma(i,j)=(i==j?.9:0.)+.002*std::exp(-.03*std::abs(i-j)); }
  return {gamma,mu,sigma,.7,1.15,.43};
}
Matrix make_rhs_raw_column_major(int n,int columns) {
  Matrix rhs(n,columns);
  // The test wrapper deliberately accepts raw column-major action columns.
  for(int col=0;col<columns;++col) for(int row=0;row<n;++row) rhs.a[size_t(col)*n+row]=value(row,col);
  return rhs;
}
struct Error { double abs=0.,rel=0.; };
Error compare(const Matrix&a,const Matrix&b) { Error e; for(size_t i=0;i<a.a.size();++i) { const double d=std::abs(a.a[i]-b.a[i]); e.abs=std::max(e.abs,d); e.rel=std::max(e.rel,d/std::max(1.,std::abs(a.a[i]))); } return e; }
bool same(const Matrix&a,const Matrix&b) { return a.a.size()==b.a.size() && std::memcmp(a.a.data(),b.a.data(),a.a.size()*sizeof(double))==0; }
}

int main() {
  try {
    // Mask before the first CUDA-context construction: logical zero is AF GV100.
    const std::string mask=std::string(NWorker15kFigure1Config::gv100_uuid_0)+","+NWorker15kFigure1Config::gv100_uuid_1;
    if(setenv("CUDA_VISIBLE_DEVICES",mask.c_str(),1)!=0) throw std::runtime_error("setenv CUDA_VISIBLE_DEVICES failed");
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop,0);
    if(std::string(prop.name).find("GV100")==std::string::npos) throw std::runtime_error("logical device 0 is not GV100");
    constexpr int n=37,m=64,columns=37;
    const auto worker=make_worker(n); const auto knots=make_knots(m); const auto parameters=make_parameters(m); const auto rhs=make_rhs_raw_column_major(n,columns); const auto rhs_before=rhs;
    GpuContext context(0);
    bool ok=true;
    for(int parameter=0;parameter<2;++parameter) {
      const Matrix scalar=test_theta_a_action_tile(context,worker,knots,parameters,parameter,rhs,1,false);
      for(const int width:std::vector<int>{1,7,31,32}) {
        const Matrix tiled=test_theta_a_action_tile(context,worker,knots,parameters,parameter,rhs,width,true);
        const auto e=compare(scalar,tiled); const bool pass=e.abs<=kAbsTolerance&&e.rel<=kRelTolerance&&same(rhs,rhs_before);
        std::cout<<"THETA_A_TILE parameter="<<parameter<<" width="<<width<<" columns="<<columns<<" remainder="<<(columns%width)<<" MAX_ABS="<<e.abs<<" MAX_REL="<<e.rel<<" HOST_INPUT_CANARY="<<(same(rhs,rhs_before)?"PASS":"FAIL")<<" DEVICE_CANARIES=PASS RESULT="<<(pass?"PASS":"FAIL")<<'\n';
        ok=ok&&pass;
      }
      const Matrix once=test_theta_a_action_tile(context,worker,knots,parameters,parameter,rhs,32,true);
      const Matrix twice=test_theta_a_action_tile(context,worker,knots,parameters,parameter,rhs,32,true);
      const bool repeat=same(once,twice);
      std::cout<<"THETA_A_TILE_REPEAT parameter="<<parameter<<" width=32 BYTE_REPEAT="<<(repeat?"PASS":"FAIL")<<'\n'; ok=ok&&repeat;
    }
    const auto&r=context.region();
    std::cout<<"THETA_A_TILE_ALLOCATIONS="<<r.mixed_action_tile_allocation_count<<" N_CAPACITY="<<r.mixed_action_tile_n_capacity<<" M_CAPACITY="<<r.mixed_action_tile_m_capacity<<" BYTES="<<(r.mixed_action_tile_n_capacity+r.mixed_action_tile_m_capacity)*sizeof(double)*2<<'\n';
    std::cout<<"THETA_A_TILE_PARITY="<<(ok?"PASS":"FAIL")<<'\n'; return ok?0:1;
  } catch(const std::exception&e) { std::cerr<<e.what()<<'\n'; return 2; }
}
