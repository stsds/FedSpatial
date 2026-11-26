#include "fedspatial/figure1_fixture.hpp"
#include "fedspatial/gpu_context.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace fedspatial;
namespace fs = std::filesystem;

namespace {
template <class T>
void write_scalar(std::ofstream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}
void write_matrix(std::ofstream& out, const Matrix& matrix) {
  const std::uint64_t count = matrix.a.size();
  write_scalar(out, count);
  out.write(reinterpret_cast<const char*>(matrix.a.data()), count * sizeof(double));
}
GlobalParameters initial(int q, int m) {
  Matrix gamma(q, 1), mu(m, 1), sigma(m, m);
  const double values[5] = {-1, 2, 3, -2, 1};
  if (q != 5) throw std::runtime_error("fixture requires five covariates");
  for (int i = 0; i < q; ++i) gamma(i, 0) = values[i];
  return {gamma, mu, sigma, 0.25, 1.0, 0.11, 3};
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 3 && argc != 4)
      throw std::runtime_error("usage: exporter FIXTURE_DIR OUTPUT_BIN [--block12-only]");
    const bool include_block3 = argc == 3;
    Figure1Fixture fixture;
    if (!load_figure1_python_fixture(argv[1], &fixture))
      throw std::runtime_error("failed to load Figure 1 fixture");
    auto shards = partition_figure1_fixture(fixture, 1);
    GpuContext context(0);
    const auto parameters = initial(fixture.x.cols, fixture.knots.rows);
    const auto reference = compute_mu_sigma_local_legacy_oracle(
        context, shards.at(0), fixture.knots, parameters);
    const Matrix knot_covariance = matern_with_derivatives(
        context, fixture.knots, fixture.knots, parameters.nu2,
        parameters.alpha, parameters.range).value;
    Matrix global_precision = inverse_spd(knot_covariance);
    for (std::size_t i = 0; i < global_precision.a.size(); ++i)
      global_precision.a[i] += parameters.delta * reference.summary.sigma.a[i];
    const Matrix global_sigma = inverse_spd(global_precision);
    Matrix global_mu = matmul(global_sigma, reference.summary.mu);
    for (double& value : global_mu.a) value *= -parameters.delta;
    GlobalParameters block1_state = parameters;
    block1_state.mu = global_mu;
    block1_state.sigma = global_sigma;
    const auto gamma_reference = compute_gamma_local(
        context, shards.at(0), fixture.knots, block1_state);
    const Matrix updated_gamma = solve_spd(
        gamma_reference.xx, gamma_reference.gamma);
    block1_state.gamma = updated_gamma;
    Matrix delta_theta_gradient(3, 1), delta_theta_hessian(3, 3);
    DeltaThetaLocalResult delta_theta_reference{};
    if (include_block3) {
      delta_theta_reference = compute_delta_theta_local(
          context, shards.at(0), fixture.knots, block1_state);
      for (int i = 0; i < 3; ++i)
        delta_theta_gradient(i, 0) = delta_theta_reference.gradient[i];
      for (int i = 0; i < 9; ++i)
        delta_theta_hessian.a[i] = delta_theta_reference.hessian[i];
    }

    std::ofstream out(argv[2], std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to open output");
    const std::uint64_t magic = 0x465350424C4B314DULL; // FSPBLK1M
    write_scalar(out, magic);
    write_scalar(out, reference.M.rows);
    write_scalar(out, reference.B.cols);
    write_scalar(out, parameters.delta);
    write_matrix(out, reference.M);
    write_matrix(out, reference.B);
    write_matrix(out, reference.e);
    write_matrix(out, reference.chol_M);
    write_matrix(out, reference.summary.mu);
    write_matrix(out, reference.summary.sigma);
    // Appended fields keep the original payload prefix readable by older
    // probes while enabling validation of the complete Block 1 global update.
    write_matrix(out, knot_covariance);
    write_matrix(out, global_mu);
    write_matrix(out, global_sigma);
    write_matrix(out, shards.at(0).x);
    write_matrix(out, parameters.gamma);
    write_matrix(out, gamma_reference.gamma);
    write_matrix(out, gamma_reference.xx);
    write_matrix(out, updated_gamma);
    if (include_block3) {
      write_scalar(out, delta_theta_reference.objective);
      write_matrix(out, delta_theta_gradient);
      write_matrix(out, delta_theta_hessian);
    }
    if (!out) throw std::runtime_error("failed while writing output");
    std::cout << "BLOCK1_FIXTURE_EXPORT n=" << reference.M.rows
              << " m=" << reference.B.cols << " delta=" << parameters.delta << "\n"
              << "BLOCK1_FIXTURE_EXPORT_STATUS=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "BLOCK1_FIXTURE_EXPORT_FAILURE=" << e.what() << "\n";
    return 2;
  }
}
