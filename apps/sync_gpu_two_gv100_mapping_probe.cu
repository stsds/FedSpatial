#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cctype>
#include <thread>

__global__ void burn(unsigned long long* value) { unsigned long long x=threadIdx.x+1; for(int i=0;i<100000000;i++)x=x*1664525ULL+1013904223ULL; value[threadIdx.x]=x; }
int main() {
  setenv("CUDA_VISIBLE_DEVICES","GPU-737f3c7a-bf62-20b7-01f8-b3fb1a3c86f4,GPU-a7c40d19-eb27-1d13-257a-5949f98fc947",1);
  int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess||count!=2)return 2;
  const char* buses[]={"0000:AF:00.0","0000:D8:00.0"};
  for(int d=0;d<2;d++){char bus[32]{},wanted[32]{};std::strncpy(wanted,buses[d],sizeof(wanted)-1);cudaDeviceProp p{};if(cudaGetDeviceProperties(&p,d)||cudaDeviceGetPCIBusId(bus,sizeof(bus),d))return 3;for(char* q: {bus,wanted})for(;*q;++q)*q=std::tolower(*q);std::cerr<<"MAPPING logical="<<d<<" bus="<<bus<<" expected="<<wanted<<" name="<<p.name<<std::endl;if(std::strcmp(bus,wanted)||!std::strstr(p.name,"GV100"))return 3;}
  std::thread t[] = {std::thread([]{cudaSetDevice(0);unsigned long long*x;cudaMalloc(&x,256*sizeof(*x));burn<<<1,256>>>(x);cudaDeviceSynchronize();cudaFree(x);}),std::thread([]{cudaSetDevice(1);unsigned long long*x;cudaMalloc(&x,256*sizeof(*x));burn<<<1,256>>>(x);cudaDeviceSynchronize();cudaFree(x);})};
  t[0].join();t[1].join();std::cout<<"TWO_GV100_MAPPING_PROBE=PASS\n";
}
