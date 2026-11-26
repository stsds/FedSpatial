#pragma once

#include <mpi.h>
#include <stdexcept>

namespace fedspatial {

// Hierarchical MPI topology for multi-GPU federated workers.
// Ranks [j*gpus_per_worker, (j+1)*gpus_per_worker) cooperate on worker j.
// Only worker_rank==0 participates in the leaders communicator used for
// federated aggregation; worker_comm is used by cuSOLVERMp/cuBLASMp.
struct MultiGpuWorkerGroup {
  MPI_Comm world{MPI_COMM_NULL};
  MPI_Comm worker_comm{MPI_COMM_NULL};
  MPI_Comm leaders_comm{MPI_COMM_NULL};
  int world_rank{};
  int world_size{};
  int worker_id{};
  int worker_rank{};
  int worker_size{};
  int worker_count{};
  int leader_rank{-1};
  bool is_leader{};

  static MultiGpuWorkerGroup create(MPI_Comm world_comm, int gpus_per_worker) {
    if (gpus_per_worker <= 0) throw std::runtime_error("gpus_per_worker must be positive");
    MultiGpuWorkerGroup out;
    out.world = world_comm;
    MPI_Comm_rank(world_comm, &out.world_rank);
    MPI_Comm_size(world_comm, &out.world_size);
    if (out.world_size % gpus_per_worker != 0)
      throw std::runtime_error("MPI size must be divisible by gpus_per_worker");
    out.worker_count = out.world_size / gpus_per_worker;
    out.worker_id = out.world_rank / gpus_per_worker;
    MPI_Comm_split(world_comm, out.worker_id, out.world_rank, &out.worker_comm);
    MPI_Comm_rank(out.worker_comm, &out.worker_rank);
    MPI_Comm_size(out.worker_comm, &out.worker_size);
    out.is_leader = out.worker_rank == 0;
    MPI_Comm_split(world_comm, out.is_leader ? 0 : MPI_UNDEFINED,
                   out.worker_id, &out.leaders_comm);
    if (out.is_leader) MPI_Comm_rank(out.leaders_comm, &out.leader_rank);
    return out;
  }

  // Inter-worker collectives are intentionally leader-only.  This prevents
  // every GPU rank in a logical worker from contributing the same summary.
  void allreduce_leaders_sum(const double* send, double* receive, int count) const {
    if (!is_leader || leaders_comm == MPI_COMM_NULL)
      throw std::runtime_error("leader all-reduce called by a non-leader rank");
    if (MPI_Allreduce(send, receive, count, MPI_DOUBLE, MPI_SUM, leaders_comm) != MPI_SUCCESS)
      throw std::runtime_error("MPI_Allreduce failed on leaders communicator");
  }

  // After a leader updates global parameters, distribute them to the other
  // GPUs that cooperate on the same logical worker.
  void broadcast_within_worker(double* values, int count) const {
    if (MPI_Bcast(values, count, MPI_DOUBLE, 0, worker_comm) != MPI_SUCCESS)
      throw std::runtime_error("MPI_Bcast failed on worker communicator");
  }

  void destroy() {
    if (leaders_comm != MPI_COMM_NULL) MPI_Comm_free(&leaders_comm);
    if (worker_comm != MPI_COMM_NULL) MPI_Comm_free(&worker_comm);
  }
};

} // namespace fedspatial
