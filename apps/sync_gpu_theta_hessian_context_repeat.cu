#include "fedspatial/gpu_context.hpp"
#include "fedspatial/nworker_15k_figure1_config.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
using namespace fedspatial;
namespace {
WorkerInput worker(){Matrix s(128,2),z(128,1),x(128,1);for(int i=0;i<128;++i){s(i,0)=.013*i;s(i,1)=.021*(i%11);z(i,0)=std::sin(.071*(i+1)*5)+.03*i;x(i,0)=1.;}return{s,z,x};}
Matrix knots(){Matrix k(64,2);for(int j=0;j<64;++j){k(j,0)=.019*j+.004;k(j,1)=.017*(j%13)+.002;}return k;}
GlobalParameters parameters(){Matrix g(1,1),mu(64,1),s(64,64);g(0,0)=.1;for(int i=0;i<64;++i){mu(i,0)=.01*std::cos(.1*i);for(int j=0;j<64;++j)s(i,j)=(i==j?.9:0.)+.002*std::exp(-.03*std::abs(i-j));}return{g,mu,s,.7,1.15,.43};}
void gates(){for(const char*n:{"FEDSPATIAL_MIXED_TRACE_TILED_ACTION","FEDSPATIAL_MIXED_CONTRACTIONS_TILED","FEDSPATIAL_THETA_FIRST_TILED","FEDSPATIAL_THETA_HESSIAN_AA_LOG_TILED","FEDSPATIAL_THETA_HESSIAN_AR_LOG_TILED","FEDSPATIAL_THETA_HESSIAN_RR_LOG_TILED","FEDSPATIAL_THETA_HESSIAN_Q_W_TILED"})setenv(n,"1",1);}
void compare(const char* tag,const DeviceDeltaThetaProducerBoundaryAudit&a,const DeviceDeltaThetaProducerBoundaryAudit&b){const int defined[]={3,4,11,11};const int finals[][3]={{0,-1,-1},{0,1,-1},{0,1,-1},{0,1,4}};for(int q=0;q<4;++q){for(int i=0;i<defined[q];++i)if(a.summary[q][i]!=b.summary[q][i]){std::cout<<"BOUNDARY_FIRST_DIFF "<<tag<<" boundary="<<q<<" record=summary slot="<<i<<" a="<<a.summary[q][i]<<" b="<<b.summary[q][i]<<'\n';return;}for(int k=0;k<3&&finals[q][k]>=0;++k){const int i=finals[q][k];if(a.final[q][i]!=b.final[q][i]){std::cout<<"BOUNDARY_FIRST_DIFF "<<tag<<" boundary="<<q<<" record=final slot="<<i<<" a="<<a.final[q][i]<<" b="<<b.final[q][i]<<'\n';return;}}}std::cout<<"BOUNDARY_FIRST_DIFF "<<tag<<" NONE_BYTE_EQUAL\n";}
}
int main(){try{setenv("CUDA_VISIBLE_DEVICES",NWorker15kFigure1Config::gv100_uuid_0,1);gates();const auto w=worker();const auto k=knots();const auto p=parameters();GpuContext c(0);const auto cold=compute_device_delta_theta_producer_boundary_audit_for_test(c,w,k,p);const auto warm=compute_device_delta_theta_producer_boundary_audit_for_test(c,w,k,p);GpuContext f(0);const auto fresh=compute_device_delta_theta_producer_boundary_audit_for_test(f,w,k,p);compare("cold_warm",cold,warm);compare("cold_fresh",cold,fresh);compare("warm_fresh",warm,fresh);return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
