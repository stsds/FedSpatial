#include "fedspatial/gpu_context.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace fs=std::filesystem; using namespace fedspatial;
static Matrix load(const fs::path&p){std::ifstream f(p);int r,c;if(!(f>>r>>c))throw std::runtime_error("missing "+p.string());Matrix x(r,c);for(double&v:x.a)f>>v;return x;}
static WorkerInput worker3(const fs::path& root){const auto all=load(root/"worker_0.txt");Matrix s(3,2),z(3,1),x(3,5);for(int i=0;i<3;++i){s(i,0)=all(i,0);s(i,1)=all(i,1);z(i,0)=all(i,2);for(int j=0;j<5;++j)x(i,j)=all(i,j+3);}return{s,z,x};}
int main(){try{
  const fs::path root="sync_gpu/tests/fixtures/figure1_reference"; const auto k=load(root/"knots.txt");
  GlobalParameters p{load(root/"p_initial_gamma_new.txt"),load(root/"p_initial_global_mu.txt"),load(root/"p_initial_global_sigma.txt"),.25,1.,.11};
  const auto w=worker3(root); GpuContext c(0);
  const auto a=compute_device_delta_theta_producer_delta_delta_audit_for_test(c,w,k,p);
  const auto full=compute_device_delta_theta_local(c,w,k,p).summary;
  const auto oracle=compute_delta_theta_local_legacy_oracle(c,w,k,p,make_p_global_derivative_workspace(c,k,p));
  const auto terms=[&]{const auto&q=a.sources;const double d=p.delta,d2=d*d;const double u2=q[2]+q[3]/d-q[6];const double hd=-((q[0]-u2)/d+(q[1]-q[6])/d);const double ldd=(2.*q[3]-q[4])/d2;const double hdd=2.*(q[0]-2.*u2+q[5]+q[1]-2.*q[6]+q[7])/d2;return std::array<double,4>{ldd,2.*hd,d*hdd,ldd+2.*hd+d*hdd};}();
  const double source_assembly=terms[3];
  const double standalone_err=std::abs(a.standalone_delta_hessian-oracle.hessian[0]);
  const double source_err=std::abs(source_assembly-a.device_assembled_delta_hessian);
  const double final_err=std::abs(full.hessian[0]-a.device_assembled_delta_hessian);
  std::cout<<std::setprecision(17)<<"PRODUCER_DD_N=3 STANDALONE="<<a.standalone_delta_hessian<<" DEVICE_ASSEMBLED="<<a.device_assembled_delta_hessian<<" FINAL4="<<full.hessian[0]<<" WANT="<<oracle.hessian[0]<<'\n';
  const char* names[]={"BASE_QUADRATIC","BASE_TRACE_Q","DELTA_SCORE","TR_W","TR_W2","U_W_U","TR_W2_Q","TR_W3_Q"};
  for(int i=0;i<8;++i)std::cout<<"PRODUCER_DD_SOURCE_"<<names[i]<<'='<<a.sources[i]<<'\n';
  std::cout<<"PRODUCER_DD_TERMS LDD="<<terms[0]<<" TWO_HD="<<terms[1]<<" DELTA_HDD="<<terms[2]<<" SUM="<<terms[3]<<'\n';
  std::cout<<"PRODUCER_DD_STANDALONE_ABS="<<standalone_err<<" SOURCE_ASSEMBLY_ABS="<<source_err<<" FINAL_SLOT_ABS="<<final_err<<'\n'
           <<"PRODUCER_DD_PARITY="<<(standalone_err<=1e-9&&source_err<=1e-12&&final_err==0.?"PASS":"FAIL")<<'\n';
  return standalone_err<=1e-9&&source_err<=1e-12&&final_err==0.?0:1;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
