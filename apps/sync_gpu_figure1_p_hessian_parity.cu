#include "fedspatial/gpu_context.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace fedspatial;
namespace fs = std::filesystem;

namespace {
Matrix load(const fs::path& path) {
  Matrix matrix;
  if (!read_matrix(path.string(), &matrix)) throw std::runtime_error("missing " + path.string());
  return matrix;
}

WorkerInput worker_from_fixture(const fs::path& root, int worker_id) {
  const Matrix data = load(root / ("worker_" + std::to_string(worker_id) + ".txt"));
  WorkerInput worker{Matrix(data.rows, 2), Matrix(data.rows, 1), Matrix(data.rows, data.cols - 3)};
  for (int row = 0; row < data.rows; ++row) {
    worker.sites(row, 0) = data(row, 0);
    worker.sites(row, 1) = data(row, 1);
    worker.z(row, 0) = data(row, 2);
    for (int column = 0; column < worker.x.cols; ++column) worker.x(row, column) = data(row, column + 3);
  }
  return worker;
}

void write_row(std::ofstream& csv, int worker, const char* quantity, int i, int j,
               double python_value, double cpp_value, double& maximum) {
  const double error = std::abs(python_value - cpp_value);
  maximum = std::max(maximum, error);
  csv << worker << ',' << quantity << ',' << i << ',' << j << ','
      << std::setprecision(17) << python_value << ',' << cpp_value << ',' << error << ','
      << error / std::max(1e-300, std::abs(python_value)) << '\n';
}

GlobalParameters with_theta(const GlobalParameters& p, int coordinate, double value) {
  GlobalParameters out = p;
  if (coordinate == 0) out.delta = value;
  if (coordinate == 1) out.alpha = value;
  if (coordinate == 2) out.range = value;
  return out;
}
}  // namespace

int main() {
  try {
    const fs::path fixture = "sync_gpu/tests/fixtures/figure1_reference";
    const fs::path output = "results/figure1_cpp_gpu_faithful/parity";
    fs::create_directories(output);
    const Matrix knots = load(fixture / "knots.txt");
    const Matrix mu = load(fixture / "p_initial_global_mu.txt");
    const Matrix sigma = load(fixture / "p_initial_global_sigma.txt");
    const Matrix gamma = load(fixture / "p_initial_gamma_new.txt");
    GlobalParameters parameters{gamma, mu, sigma, .25, 1., .11};
    GpuContext context(0);
    std::ofstream csv(output / "p_gradient_hessian_parity.csv");
    csv << "worker,quantity,i,j,python_value,cpp_value,abs_error,rel_error\n";
    double objective_error = 0.0;
    double gradient_error = 0.0;
    double hessian_error = 0.0;
    std::array<double, 6> hessian_component_error{};
    int worst_worker = -1;
    int worst_i = -1;
    int worst_j = -1;
    for (int worker_id = 0; worker_id < 10; ++worker_id) {
      const WorkerInput worker = worker_from_fixture(fixture, worker_id);
      const auto result = compute_delta_theta_local(context, worker, knots, parameters);
      const Matrix python_objective = load(fixture / ("p_initial_worker_" + std::to_string(worker_id) + "_objective.txt"));
      const Matrix python_gradient = load(fixture / ("p_initial_worker_" + std::to_string(worker_id) + "_gradient.txt"));
      const Matrix python_hessian = load(fixture / ("p_initial_worker_" + std::to_string(worker_id) + "_hessian.txt"));
      write_row(csv, worker_id, "objective", 0, 0, python_objective(0, 0), result.objective, objective_error);
      for (int i = 0; i < 3; ++i) {
        write_row(csv, worker_id, "gradient", i, 0, python_gradient(i, 0), result.gradient[i], gradient_error);
        for (int j = 0; j < 3; ++j) {
          const double error = std::abs(python_hessian(i, j) - result.hessian[3 * i + j]);
          if (error > hessian_error) { worst_worker = worker_id; worst_i = i; worst_j = j; }
          write_row(csv, worker_id, "hessian", i, j, python_hessian(i, j), result.hessian[3 * i + j], hessian_error);
        }
      }
      const int component[] = {0, 1, 2, 4, 5, 8};
      for (int index = 0; index < 6; ++index) {
        hessian_component_error[index] = std::max(
            hessian_component_error[index],
            std::abs(python_hessian.a[component[index]] - result.hessian[component[index]]));
      }
    }

    // Independent native finite-difference check on worker zero.
    const WorkerInput worker = worker_from_fixture(fixture, 0);
    const auto analytic = compute_delta_theta_local(context, worker, knots, parameters);
    std::ofstream fd_csv(output / "p_finite_difference_check.csv");
    fd_csv << "epsilon,quantity,i,j,analytic,finite_difference,abs_error\n";
    for (double base_epsilon : {1e-5, 1e-6}) {
      double gradient_fd_error = 0.0;
      double hessian_fd_error = 0.0;
      for (int i = 0; i < 3; ++i) {
        const double theta = i == 0 ? parameters.delta : (i == 1 ? parameters.alpha : parameters.range);
        const double epsilon = base_epsilon * std::max(1.0, std::abs(theta));
        const auto plus = compute_delta_theta_local(context, worker, knots, with_theta(parameters, i, theta + epsilon));
        const auto minus = compute_delta_theta_local(context, worker, knots, with_theta(parameters, i, theta - epsilon));
        const double fd_gradient = (plus.objective - minus.objective) / (2.0 * epsilon);
        const double gradient_error_i = std::abs(fd_gradient - analytic.gradient[i]);
        gradient_fd_error = std::max(gradient_fd_error, gradient_error_i);
        fd_csv << std::setprecision(17) << base_epsilon << ",gradient," << i << ",0,"
               << analytic.gradient[i] << "," << fd_gradient << "," << gradient_error_i << "\n";
        for (int j = 0; j < 3; ++j) {
          const double fd_hessian = (plus.gradient[j] - minus.gradient[j]) / (2.0 * epsilon);
          const double hessian_error_ji = std::abs(fd_hessian - analytic.hessian[3 * j + i]);
          hessian_fd_error = std::max(hessian_fd_error, hessian_error_ji);
          fd_csv << std::setprecision(17) << base_epsilon << ",hessian," << j << "," << i << ","
                 << analytic.hessian[3 * j + i] << "," << fd_hessian << "," << hessian_error_ji << "\n";
        }
      }
      std::cout << "FD_BASE_EPS=" << base_epsilon
                << " FD_GRADIENT_MAX_ABS_ERROR=" << gradient_fd_error
                << " FD_HESSIAN_MAX_ABS_ERROR=" << hessian_fd_error << '\n';
    }
    std::cout << "P_LOCAL_OBJECTIVE_MAX_ABS_ERROR=" << objective_error << '\n'
              << "P_LOCAL_GRADIENT_MAX_ABS_ERROR=" << gradient_error << '\n'
              << "P_HESSIAN_DD_MAX_ABS_ERROR=" << hessian_component_error[0] << '\n'
              << "P_HESSIAN_DA_MAX_ABS_ERROR=" << hessian_component_error[1] << '\n'
              << "P_HESSIAN_DB_MAX_ABS_ERROR=" << hessian_component_error[2] << '\n'
              << "P_HESSIAN_AA_MAX_ABS_ERROR=" << hessian_component_error[3] << '\n'
              << "P_HESSIAN_AB_MAX_ABS_ERROR=" << hessian_component_error[4] << '\n'
              << "P_HESSIAN_BB_MAX_ABS_ERROR=" << hessian_component_error[5] << '\n'
              << "P_LOCAL_HESSIAN_MAX_ABS_ERROR=" << hessian_error << '\n'
              << "P_LOCAL_HESSIAN_WORST=worker_" << worst_worker << "_" << worst_i << "_" << worst_j << '\n'
              << "P_LOCAL_HESSIAN_PARITY=" << ((objective_error <= 1e-10 && gradient_error <= 1e-10 && hessian_error <= 1e-9) ? "PASS" : "FAIL") << '\n';
    return (objective_error <= 1e-10 && gradient_error <= 1e-10 && hessian_error <= 1e-9) ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
