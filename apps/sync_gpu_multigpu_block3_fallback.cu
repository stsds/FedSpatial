#include "fedspatial/figure1_fixture.hpp"
#include "fedspatial/gpu_context.hpp"
#include "fedspatial/multi_gpu_worker_group.hpp"

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace fedspatial;

namespace {
GlobalParameters initial(int q, int m) {
  Matrix gamma(q, 1), mu(m, 1), sigma(m, m);
  const double values[5] = {-1, 2, 3, -2, 1};
  if (q != 5) throw std::runtime_error("fixture requires five covariates");
  for (int i = 0; i < q; ++i) gamma(i, 0) = values[i];
  return {gamma, mu, sigma, 0.25, 1.0, 0.11, 3};
}

GlobalParameters advance_blocks_1_and_2(GpuContext& context,
                                         const WorkerInput& input,
                                         const Matrix& knots,
                                         GlobalParameters state) {
  const auto local = compute_mu_sigma_local_legacy_oracle(
      context, input, knots, state).summary;
  Matrix precision = inverse_spd(matern_with_derivatives(
      context, knots, knots, state.nu2, state.alpha, state.range).value);
  for (std::size_t i = 0; i < precision.a.size(); ++i)
    precision.a[i] += state.delta * local.sigma.a[i];
  state.sigma = inverse_spd(precision);
  state.mu = matmul(state.sigma, local.mu);
  for (double& value : state.mu.a) value *= -state.delta;
  const auto gamma = compute_gamma_local(context, input, knots, state);
  state.gamma = solve_spd(gamma.xx, gamma.gamma);
  return state;
}

struct Eig3 {
  std::array<double, 3> values{};
  Matrix vectors{3, 3};
};

Eig3 symmetric_eig3(Matrix matrix) {
  Eig3 result;
  for (int i = 0; i < 3; ++i) result.vectors(i, i) = 1.0;
  for (int iteration = 0; iteration < 64; ++iteration) {
    int p = 0, q = 1;
    double largest = std::abs(matrix(0, 1));
    for (int i = 0; i < 3; ++i)
      for (int j = i + 1; j < 3; ++j)
        if (std::abs(matrix(i, j)) > largest) {
          largest = std::abs(matrix(i, j)); p = i; q = j;
        }
    if (largest < 1e-15) break;
    const double angle = 0.5 * std::atan2(
        2.0 * matrix(p, q), matrix(q, q) - matrix(p, p));
    const double c = std::cos(angle), s = std::sin(angle);
    for (int k = 0; k < 3; ++k) {
      const double u = matrix(p, k), v = matrix(q, k);
      matrix(p, k) = c * u - s * v;
      matrix(q, k) = s * u + c * v;
    }
    for (int k = 0; k < 3; ++k) {
      double u = matrix(k, p), v = matrix(k, q);
      matrix(k, p) = c * u - s * v;
      matrix(k, q) = s * u + c * v;
      u = result.vectors(k, p); v = result.vectors(k, q);
      result.vectors(k, p) = c * u - s * v;
      result.vectors(k, q) = s * u + c * v;
    }
  }
  std::array<int, 3> order{0, 1, 2};
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return matrix(a, a) < matrix(b, b);
  });
  const Matrix unsorted = result.vectors;
  for (int k = 0; k < 3; ++k) {
    result.values[k] = matrix(order[k], order[k]);
    for (int i = 0; i < 3; ++i)
      result.vectors(i, k) = unsorted(i, order[k]);
  }
  return result;
}

struct NewtonResult {
  std::array<double, 3> parameters{};
  double step{1.0};
  int halvings{};
};

NewtonResult newton_update(GpuContext& context, const Matrix& knots,
                           const GlobalParameters& state,
                           const std::array<double, 13>& aggregate) {
  Matrix gradient(3, 1), hessian(3, 3), correction_gradient(3, 1),
      correction_hessian(3, 3);
  for (int i = 0; i < 3; ++i) gradient(i, 0) = aggregate[1 + i];
  for (int i = 0; i < 9; ++i) hessian.a[i] = aggregate[4 + i];
  compute_legacy_global_k_correction_for_test(
      context, knots, state, &correction_gradient, &correction_hessian);
  for (int i = 0; i < 3; ++i) gradient(i, 0) += correction_gradient(i, 0);
  for (int i = 0; i < 9; ++i) hessian.a[i] += correction_hessian.a[i];
  const auto eigen = symmetric_eig3(hessian);
  Matrix modified_hessian(3, 3);
  for (int k = 0; k < 3; ++k) {
    const double value = std::max(std::abs(eigen.values[k]), 0.01);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        modified_hessian(i, j) +=
            eigen.vectors(i, k) * value * eigen.vectors(j, k);
  }
  const Matrix direction = solve_spd(modified_hessian, gradient);
  NewtonResult result;
  do {
    result.parameters = {
        state.delta - result.step * direction(0, 0),
        state.alpha - result.step * direction(1, 0),
        state.range - result.step * direction(2, 0)};
    if (result.parameters[0] > 0.0 && result.parameters[1] > 0.0 &&
        result.parameters[2] > 0.0) break;
    result.step *= 0.5;
    ++result.halvings;
  } while (true);
  return result;
}
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  try {
    if (argc < 3 || argc > 5)
      throw std::runtime_error(
          "usage: block3_fallback FIXTURE_DIR GPUS_PER_WORKER [BLOCK12_STATE] [--skip-oracle]");
    const int gpus_per_worker = std::stoi(argv[2]);
    const bool verify_oracle = argc < 5 || std::string(argv[4]) != "--skip-oracle";
    auto topology = MultiGpuWorkerGroup::create(MPI_COMM_WORLD, gpus_per_worker);

    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                        MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
    if (cudaSetDevice(local_rank) != cudaSuccess)
      throw std::runtime_error("cudaSetDevice failed");

    Figure1Fixture fixture;
    if (!load_figure1_python_fixture(argv[1], &fixture))
      throw std::runtime_error("fixture load failed");
    const auto partitions = partition_figure1_fixture(
        fixture, topology.worker_count);
    const auto& input = partitions.at(topology.worker_id);
    GpuContext context(local_rank);
    auto state = initial(fixture.x.cols, fixture.knots.rows);
    if (argc == 4) {
      std::ifstream in(argv[3], std::ios::binary);
      std::uint64_t magic{};
      int m = 0, p = 0;
      in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
      in.read(reinterpret_cast<char*>(&m), sizeof(m));
      in.read(reinterpret_cast<char*>(&p), sizeof(p));
      if (magic != 0x4653505354415445ULL || m != fixture.knots.rows ||
          p != fixture.x.cols)
        throw std::runtime_error("invalid Block 1/2 state header");
      in.read(reinterpret_cast<char*>(state.mu.a.data()),
              state.mu.a.size() * sizeof(double));
      in.read(reinterpret_cast<char*>(state.sigma.a.data()),
              state.sigma.a.size() * sizeof(double));
      in.read(reinterpret_cast<char*>(state.gamma.a.data()),
              state.gamma.a.size() * sizeof(double));
      if (!in) throw std::runtime_error("truncated Block 1/2 state");
    } else {
      state = advance_blocks_1_and_2(context, input, fixture.knots, state);
    }

    MPI_Barrier(topology.worker_comm);
    const double block3_start = MPI_Wtime();
    // Split the six independent Block 3 contraction families across the
    // GPUs in this logical worker.  Every rank contributes disjoint transport
    // slots, so the fixed-order all-reduction reconstructs the 13-value
    // objective/gradient/Hessian record without double counting.
    std::array<double, 13> partial{};
    std::array<double, 6> family_seconds{};
    auto owns = [&](int family) {
      return family % topology.worker_size == topology.worker_rank;
    };
    if (owns(0)) {
      const double begin = MPI_Wtime();
      const auto value = compute_device_delta_theta_base_reduction_streamed(
          context, input, fixture.knots, state);
      partial[0] = value.objective;
      family_seconds[0] = MPI_Wtime() - begin;
    }
    if (owns(1)) {
      const double begin = MPI_Wtime();
      const auto value = compute_device_delta_theta_delta_first_reduction_streamed(
          context, input, fixture.knots, state);
      partial[1] = value.delta_gradient;
      family_seconds[1] = MPI_Wtime() - begin;
    }
    if (owns(2)) {
      const double begin = MPI_Wtime();
      const auto value = compute_device_delta_theta_delta_hessian_reduction_streamed(
          context, input, fixture.knots, state);
      partial[4] = value.delta_hessian;
      family_seconds[2] = MPI_Wtime() - begin;
    }
    for (int parameter = 0; parameter < 2; ++parameter) {
      const int task = parameter;
      if (task % topology.worker_size != topology.worker_rank) continue;
      const double begin = MPI_Wtime();
      const std::string filter = std::to_string(parameter);
      setenv("FEDSPATIAL_MIXED_PARAMETER", filter.c_str(), 1);
      const auto value = compute_device_delta_theta_mixed_hessian_reduction_streamed(
          context, input, fixture.knots, state);
      unsetenv("FEDSPATIAL_MIXED_PARAMETER");
      if (parameter == 0) partial[5] = partial[7] = value.mixed[0];
      else partial[6] = partial[10] = value.mixed[1];
      family_seconds[3] += MPI_Wtime() - begin;
    }
    for (int parameter = 0; parameter < 2; ++parameter) {
      const int task = 2 + parameter;
      if (task % topology.worker_size != topology.worker_rank) continue;
      const double begin = MPI_Wtime();
      const std::string filter = std::to_string(parameter);
      setenv("FEDSPATIAL_THETA_FIRST_PARAMETER", filter.c_str(), 1);
      const auto value = compute_device_delta_theta_theta_first_reduction_streamed(
          context, input, fixture.knots, state);
      unsetenv("FEDSPATIAL_THETA_FIRST_PARAMETER");
      if (parameter == 0) partial[2] = value.alpha_gradient;
      else partial[3] = value.range_gradient;
      family_seconds[4] += MPI_Wtime() - begin;
    }
    for (int entry = 0; entry < 3; ++entry) {
      const int task = 4 + entry;
      if (task % topology.worker_size != topology.worker_rank) continue;
      const double begin = MPI_Wtime();
      const std::string filter = std::to_string(entry);
      setenv("FEDSPATIAL_THETA_HESSIAN_ENTRY", filter.c_str(), 1);
      const auto value = compute_device_delta_theta_theta_hessian_reduction_streamed(
          context, input, fixture.knots, state);
      unsetenv("FEDSPATIAL_THETA_HESSIAN_ENTRY");
      if (entry == 0) partial[8] = value.hessian[0];
      else if (entry == 1) partial[9] = partial[11] = value.hessian[1];
      else partial[12] = value.hessian[2];
      family_seconds[5] += MPI_Wtime() - begin;
    }

    std::array<double, 13> summary{};
    MPI_Allreduce(partial.data(), summary.data(), int(summary.size()),
                  MPI_DOUBLE, MPI_SUM, topology.worker_comm);
    std::array<double, 6> worker_family_seconds{};
    MPI_Allreduce(family_seconds.data(), worker_family_seconds.data(),
                  int(family_seconds.size()), MPI_DOUBLE, MPI_SUM,
                  topology.worker_comm);
    const double local_block3_seconds = MPI_Wtime() - block3_start;
    double block3_seconds = 0.0;
    MPI_Allreduce(&local_block3_seconds, &block3_seconds, 1, MPI_DOUBLE,
                  MPI_MAX, topology.worker_comm);
    std::array<double, 13> oracle{};
    if (topology.is_leader && verify_oracle) {
      const auto global_workspace = make_p_global_derivative_workspace(
          context, fixture.knots, state);
      const auto reference = compute_delta_theta_local_legacy_oracle(
          context, input, fixture.knots, state, global_workspace);
      oracle[0] = reference.objective;
      for (int i = 0; i < 3; ++i) {
        oracle[1 + i] = reference.gradient[i];
      }
      for (int i = 0; i < 9; ++i) {
        oracle[4 + i] = reference.hessian[i];
      }
    }
    if (!verify_oracle) oracle = summary;
    topology.broadcast_within_worker(oracle.data(), int(oracle.size()));
    std::array<double, 13> global_summary{}, global_oracle{};
    if (topology.is_leader) {
      topology.allreduce_leaders_sum(summary.data(), global_summary.data(),
                                     int(summary.size()));
      topology.allreduce_leaders_sum(oracle.data(), global_oracle.data(),
                                     int(oracle.size()));
    }
    topology.broadcast_within_worker(global_summary.data(),
                                     int(global_summary.size()));
    topology.broadcast_within_worker(global_oracle.data(),
                                     int(global_oracle.size()));
    double local_error = 0.0;
    for (std::size_t i = 0; i < summary.size(); ++i)
      local_error = std::max(local_error, std::abs(summary[i] - oracle[i]));
    double parameter_error = 0.0;
    int decision_mismatch = 0;
    if (topology.is_leader) {
      const auto distributed = newton_update(
          context, fixture.knots, state, global_summary);
      const auto reference = newton_update(
          context, fixture.knots, state, global_oracle);
      for (int i = 0; i < 3; ++i)
        parameter_error = std::max(
            parameter_error,
            std::abs(distributed.parameters[i] - reference.parameters[i]));
      decision_mismatch = distributed.halvings != reference.halvings ||
                          distributed.step != reference.step;
    }
    MPI_Bcast(&parameter_error, 1, MPI_DOUBLE, 0, topology.worker_comm);
    MPI_Bcast(&decision_mismatch, 1, MPI_INT, 0, topology.worker_comm);
    double max_error = 0.0;
    MPI_Allreduce(&local_error, &max_error, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    double max_parameter_error = 0.0;
    int any_decision_mismatch = 0;
    MPI_Allreduce(&parameter_error, &max_parameter_error, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&decision_mismatch, &any_decision_mismatch, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    if (world_rank == 0) {
      std::cout << "MULTIGPU_BLOCK3_DISTRIBUTED max_abs=" << max_error
                << " parameter_max_abs=" << max_parameter_error
                << " decision_mismatch=" << any_decision_mismatch
                << " block3_seconds=" << block3_seconds
                << " family_seconds=";
      for (std::size_t i = 0; i < worker_family_seconds.size(); ++i)
        std::cout << (i ? "," : "") << worker_family_seconds[i];
      std::cout << '\n';
      if (max_error > 1e-8 || max_parameter_error > 1e-10 ||
          any_decision_mismatch)
        throw std::runtime_error("distributed Block 3 parity failed");
      std::cout << "MULTIGPU_BLOCK3_DISTRIBUTED_STATUS=PASS\n";
    }
    topology.destroy();
    MPI_Finalize();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rank=" << world_rank << " ERROR=" << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 2);
    return 2;
  }
}
