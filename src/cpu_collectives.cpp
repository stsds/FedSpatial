#include "fedspatial/cpu_backend.hpp"
#include <cstdint>
#include <limits>
#include <stdexcept>
#ifdef FEDSPATIAL_CPU_MPI
#include <mpi.h>
#endif
namespace fedspatial {
namespace {
thread_local int rank=0,size=1;
thread_local std::uint64_t communication_bytes=0;
#ifdef FEDSPATIAL_CPU_MPI
thread_local MPI_Comm comm=MPI_COMM_NULL;
#endif
}
void cpu_reset_communication_bytes(){communication_bytes=0;}
void cpu_add_communication_bytes(std::uint64_t bytes){communication_bytes+=bytes;}
std::uint64_t cpu_communication_bytes(){return communication_bytes;}
int cpu_row_begin(int rows){return int(static_cast<long long>(rows)*rank/size);}
int cpu_row_end(int rows){return int(static_cast<long long>(rows)*(rank+1)/size);}
#ifdef FEDSPATIAL_CPU_MPI
void cpu_set_worker_comm(int r,int s,const void* c){rank=r;size=s;comm=c?*static_cast<const MPI_Comm*>(c):MPI_COMM_NULL;}
#endif
void cpu_gather_rows(Matrix& a) {
#ifdef FEDSPATIAL_CPU_MPI
  if(size==1)return;
  if(a.a.size()>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("CPU collective exceeds MPI count limit");
  std::vector<int> counts(size),displacements(size);
  for(int r=0;r<size;++r){int b=int(static_cast<long long>(a.rows)*r/size),e=int(static_cast<long long>(a.rows)*(r+1)/size);counts[r]=(e-b)*a.cols;displacements[r]=b*a.cols;}
  communication_bytes+=std::uint64_t(counts[rank])*sizeof(double)*(size-1);
  if(MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DOUBLE,a.a.data(),counts.data(),displacements.data(),MPI_DOUBLE,comm)!=MPI_SUCCESS)
    throw std::runtime_error("CPU intra-worker row gather failed");
#else
  (void)a;
#endif
}
}
