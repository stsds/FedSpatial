#include "fedspatial/nworker_15k_figure1_config.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
void ck(cudaError_t e,const char* w){if(e!=cudaSuccess)throw std::runtime_error(std::string(w)+": "+cudaGetErrorString(e));}
void cb(cublasStatus_t s,const char* w){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(w)+": cublas status "+std::to_string(int(s)));}
double v(int i){return std::sin(.071*(i+1))+.003*i;}
}
int main(){try{
  setenv("CUDA_VISIBLE_DEVICES",fedspatial::NWorker15kFigure1Config::gv100_uuid_0,1);
  cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0),"device properties");if(std::string(prop.name).find("GV100")==std::string::npos)throw std::runtime_error("logical device 0 is not GV100");
  constexpr int n=128,m=64,triples=9,replays=384; const size_t nm=size_t(n)*m,mm=size_t(m)*m;
  std::vector<double> h_right(nm),h_left(nm),h_middle(mm),h_x(n),h_base(n),reference(n),graph_out(n);
  for(size_t i=0;i<nm;++i){h_right[i]=v(int(i%97));h_left[i]=v(int((3*i)%101));}for(size_t i=0;i<mm;++i)h_middle[i]=(i%m==i/m?.8:0.)+.001*v(int(i));for(int i=0;i<n;++i){h_x[i]=v(i+5);h_base[i]=v(i+17);}
  double *right{},*left{},*middle{},*x{},*base{},*out{},*tmp{},*v0{},*v1{};for(auto p:{&right,&left})ck(cudaMalloc(p,nm*sizeof(double)),"matrix alloc");for(auto p:{&middle})ck(cudaMalloc(p,mm*sizeof(double)),"middle alloc");for(auto p:{&x,&base,&out,&tmp})ck(cudaMalloc(p,n*sizeof(double)),"vector alloc");for(auto p:{&v0,&v1})ck(cudaMalloc(p,m*sizeof(double)),"m vector alloc");
  cudaStream_t stream{};ck(cudaStreamCreate(&stream),"stream");cublasHandle_t h{};cb(cublasCreate(&h),"cublas create");cb(cublasSetStream(h,stream),"cublas stream");
  ck(cudaMemcpyAsync(right,h_right.data(),nm*sizeof(double),cudaMemcpyHostToDevice,stream),"right upload");ck(cudaMemcpyAsync(left,h_left.data(),nm*sizeof(double),cudaMemcpyHostToDevice,stream),"left upload");ck(cudaMemcpyAsync(middle,h_middle.data(),mm*sizeof(double),cudaMemcpyHostToDevice,stream),"middle upload");ck(cudaMemcpyAsync(x,h_x.data(),n*sizeof(double),cudaMemcpyHostToDevice,stream),"x upload");ck(cudaMemcpyAsync(base,h_base.data(),n*sizeof(double),cudaMemcpyHostToDevice,stream),"base upload");ck(cudaStreamSynchronize(stream),"uploads");
  const double one=1.,minus=-1.,zero=0.;auto sequence=[&](){ck(cudaMemcpyAsync(out,base,n*sizeof(double),cudaMemcpyDeviceToDevice,stream),"base copy");for(int t=0;t<triples;++t){cb(cublasDgemv(h,CUBLAS_OP_N,m,n,&one,right,m,x,1,&zero,v0,1),"right gemv");cb(cublasDgemv(h,CUBLAS_OP_N,m,m,&one,middle,m,v0,1,&zero,v1,1),"middle gemv");cb(cublasDgemv(h,CUBLAS_OP_T,m,n,&one,left,m,v1,1,&zero,tmp,1),"left gemv");cb(cublasDaxpy(h,n,&minus,tmp,1,out,1),"daxpy");}};
  sequence();ck(cudaMemcpyAsync(reference.data(),out,n*sizeof(double),cudaMemcpyDeviceToHost,stream),"reference copy");ck(cudaStreamSynchronize(stream),"reference sync");
  cudaGraph_t graph{};cudaGraphExec_t exec{};const auto capture_start=std::chrono::steady_clock::now();ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal),"begin capture");sequence();ck(cudaStreamEndCapture(stream,&graph),"end capture");ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0),"instantiate");const double capture_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-capture_start).count();size_t nodes=0;ck(cudaGraphGetNodes(graph,nullptr,&nodes),"graph nodes");
  cudaEvent_t a{},b{};ck(cudaEventCreate(&a),"event a");ck(cudaEventCreate(&b),"event b");ck(cudaEventRecord(a,stream),"event record a");for(int r=0;r<replays;++r)ck(cudaGraphLaunch(exec,stream),"graph launch");ck(cudaEventRecord(b,stream),"event record b");ck(cudaEventSynchronize(b),"event sync b");float graph_ms=0;ck(cudaEventElapsedTime(&graph_ms,a,b),"event elapsed");ck(cudaMemcpyAsync(graph_out.data(),out,n*sizeof(double),cudaMemcpyDeviceToHost,stream),"graph copy");ck(cudaStreamSynchronize(stream),"graph sync");
  double ma=0.,mr=0.;for(int i=0;i<n;++i){const double d=std::abs(reference[i]-graph_out[i]);ma=std::max(ma,d);mr=std::max(mr,d/std::max({1.,std::abs(reference[i]),std::abs(graph_out[i])}));}const bool byte=std::memcmp(reference.data(),graph_out.data(),n*sizeof(double))==0;
  ck(cudaEventRecord(a,stream),"normal record a");for(int r=0;r<replays;++r)sequence();ck(cudaEventRecord(b,stream),"normal record b");ck(cudaEventSynchronize(b),"normal sync");float normal_ms=0;ck(cudaEventElapsedTime(&normal_ms,a,b),"normal elapsed");
  std::cout<<"THETA_LOG_CUDA_GRAPH GPU="<<prop.name<<" n="<<n<<" m="<<m<<" triples="<<triples<<" primitive_calls_per_sequence="<<(1+triples*4)<<" graph_nodes="<<nodes<<" capture_setup_ms="<<capture_ms<<" normal_cuda_ms="<<normal_ms<<" graph_replay_cuda_ms="<<graph_ms<<" speedup="<<normal_ms/graph_ms<<" max_abs="<<ma<<" max_rel="<<mr<<" byte_repeat="<<(byte?"PASS":"FAIL")<<" RESULT="<<(byte&&mr<=2e-13?"PASS":"FAIL")<<std::endl;
  cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);cudaEventDestroy(a);cudaEventDestroy(b);cublasDestroy(h);cudaStreamDestroy(stream);for(double*p:{right,left,middle,x,base,out,tmp,v0,v1})cudaFree(p);return byte&&mr<=2e-13?0:1;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
