#include "fedspatial/gpu_context.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fedspatial;
namespace fs = std::filesystem;

static Matrix load(const fs::path& path) {
  Matrix matrix;
  if (!read_matrix(path.string(), &matrix)) throw std::runtime_error("missing " + path.string());
  return matrix;
}

int main(int argc, char** argv) {
  try {
    const int rounds = argc > 1 ? std::stoi(argv[1]) : 50;
    const fs::path output = argc > 2 ? argv[2] : "results/figure1_cpp_gpu_faithful/profiling/p_fixture_profile.csv";
    const fs::path fixture = "sync_gpu/tests/fixtures/figure1_reference";
    Matrix knots = load(fixture / "knots.txt");
    Matrix gamma(5, 1), mu(100, 1), sigma(100, 100);
    const double gamma_values[] = {-1, 2, 3, -2, 1};
    for (int i = 0; i < 5; ++i) gamma(i, 0) = gamma_values[i];
    std::vector<WorkerInput> workers;
    for (int worker_id = 0; worker_id < 10; ++worker_id) {
      Matrix data = load(fixture / ("worker_" + std::to_string(worker_id) + ".txt"));
      WorkerInput worker{Matrix(100, 2), Matrix(100, 1), Matrix(100, 5)};
      for (int row = 0; row < 100; ++row) {
        worker.sites(row, 0) = data(row, 0);
        worker.sites(row, 1) = data(row, 1);
        worker.z(row, 0) = data(row, 2);
        for (int column = 0; column < 5; ++column) worker.x(row, column) = data(row, column + 3);
      }
      workers.push_back(std::move(worker));
    }

    GpuContext context(0);
    GlobalParameters parameters{gamma, mu, sigma, .25, 1., .11};
    context.profile().reset(true);
    const auto start = std::chrono::steady_clock::now();
    for (int round = 0; round < rounds; ++round)
      parameters = run_p_synchronous_round(context, workers, knots, parameters).parameters;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    fs::create_directories(output.parent_path());
    std::ofstream profile(output);
    profile << "total_p_seconds," << seconds << '\n';
    context.profile().write_csv(profile);
    std::cout << "P_PROFILE_ROUNDS=" << rounds << "\nP_PROFILE_SECONDS=" << seconds
              << "\nP_PROFILE_OUTPUT=" << output << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
