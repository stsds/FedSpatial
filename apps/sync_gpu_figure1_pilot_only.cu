#include "fedspatial/figure1_fixture.hpp"
#include "fedspatial/gpu_context.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

// This executable is deliberately a hard bounded target, not a benchmark
// interface: exactly three logical workers (0/1/2), one initialization, and
// exactly one synchronous FP64 P round.  It has no IND/F/H code path.
using namespace fedspatial;
namespace fs = std::filesystem;
namespace {
constexpr int kWorkers = 3, kLocalN = 15000, kTotalN = kWorkers * kLocalN;
constexpr int kM = 2000, kQ = 5;
constexpr double kModeledPeakGiB = 7.2726;
const fs::path kFixture = "results/sync_gpu_pilot_fixture/scenario_04/rep_000";

void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
bool finite(const Matrix& x) { for (double v : x.a) if (!std::isfinite(v)) return false; return true; }
bool finite(const GlobalParameters& p) {
  return finite(p.gamma) && finite(p.mu) && finite(p.sigma) && std::isfinite(p.delta) &&
         std::isfinite(p.alpha) && std::isfinite(p.range);
}
GlobalParameters initial() {
  Matrix gamma(kQ, 1), mu(kM, 1), sigma(kM, kM);
  const double values[kQ] = {-1., 2., 3., -2., 1.};
  for (int i = 0; i < kQ; ++i) gamma(i, 0) = values[i];
  for (int i = 0; i < kM; ++i) sigma(i, i) = 1.;
  return {gamma, mu, sigma, .25, 1., .11, 3};
}
void print_parameters(const char* label, const GlobalParameters& p) {
  std::cout << std::setprecision(17) << "ACCURACY_REFERENCE " << label
            << " finite=" << (finite(p) ? "PASS" : "FAIL")
            << " gamma=";
  for (int i = 0; i < kQ; ++i) std::cout << (i ? "," : "") << p.gamma(i, 0);
  std::cout << " delta=" << p.delta << " alpha=" << p.alpha << " beta=" << p.range << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    // Preserve configuration and checkpoint output if an external safety
    // timeout terminates this one-round executable mid-stage.
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    require(argc == 2 && std::string(argv[1]) == "--pilot-only",
            "refusing launch: sole invocation is --pilot-only");
    int devices = 0;
    require(cudaGetDeviceCount(&devices) == cudaSuccess && devices >= 3,
            "requires CUDA devices 0,1,2");
    const std::vector<int> device_ids{0, 1, 2};
    std::ifstream metadata_file(kFixture / "metadata.json");
    require(bool(metadata_file), "missing exact Figure-1 pilot fixture metadata");
    const std::string metadata((std::istreambuf_iterator<char>(metadata_file)), {});
    for (const char* required : {"\"scenario_id\": 4", "\"replication\": 0", "\"seed\": 1",
                                 "\"nu\": 1.5", "\"beta\": 0.11", "\"n\": 45000",
                                 "\"p\": 5", "\"knots\": 2000", "\"workers\": 3",
                                 "GPPSampleGeneratorUnitSquare.generate_obs_gp"})
      require(metadata.find(required) != std::string::npos, "Figure-1 metadata/configuration mismatch");
    const auto init_start = std::chrono::steady_clock::now();
    Figure1Fixture fixture;
    require(load_figure1_python_fixture(kFixture, &fixture), "cannot load exact FP64 fixture");
    require(fixture.sites.rows == kTotalN && fixture.knots.rows == kM && fixture.x.cols == kQ,
            "fixture dimensions must be J=3, n_j=15000, m=2000, q=5");
    auto workers = partition_figure1_fixture(fixture, kWorkers);
    for (const auto& w : workers) require(w.sites.rows == kLocalN && finite(w.sites) && finite(w.z) && finite(w.x), "nonfinite/local-size input error");
    auto p = initial();
    NWorkerPSynchronousExecutor executor(device_ids);
    const double initialization_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << std::setprecision(17)
              << "PILOT_CONFIGURATION mode=pilot-only precision=FP64 method=P synchronous=YES rounds=1 J=3 devices=0,1,2 workers_per_gpu=1 n_j=15000 m=2000 q=5 scenario=4 replication=0 seed=1 nu=1.5 delta=0.25 alpha=1 beta=0.11 producer=COMPLETE_STREAMED_13_DOUBLE legacy_B_A_M_bundles=FORBIDDEN arena_25nn=FORBIDDEN modeled_peak_gib=" << kModeledPeakGiB << '\n'
              << "INITIALIZATION_SECONDS=" << initialization_seconds << '\n';
    print_parameters("INITIAL", p);
    // This is the one and only round call in this translation unit.
    const auto result = executor.run_round_with_checkpoints(workers, fixture.knots, p);
    require(finite(result.after_mu_sigma) && finite(result.after_gamma) && finite(result.after_delta_theta.parameters), "nonfinite round result");
    require(result.timing.raw_data_stationary, "raw-data transfer parity failure");
    const auto& t = result.timing;
    std::cout << "MU_SIGMA_SECONDS=" << t.mu_sigma_local_seconds << '\n'
              << "GAMMA_SECONDS=" << t.gamma_local_seconds << '\n'
              << "DELTA_THETA_SECONDS=" << t.delta_theta_local_seconds << '\n'
              << "AGGREGATION_SECONDS=" << t.aggregation_seconds << '\n'
              << "GLOBAL_UPDATE_SECONDS=" << t.global_update_seconds << '\n'
              << "BROADCAST_SECONDS=" << t.broadcast_seconds << '\n'
              << "ONE_ROUND_SECONDS=" << t.total_seconds << '\n'
              << "PROJECTED_50_ROUND_SECONDS=" << (50. * t.total_seconds) << '\n'
              << "H2D_RAW_BYTES=" << t.raw_h2d_bytes << " H2D_MODEL_BYTES=" << t.model_h2d_bytes
              << " D2H_MODEL_BYTES=" << t.model_d2h_bytes << " D2H_HOST_COMPUTE_BYTES=" << t.host_compute_d2h_bytes
              << " P2P_BYTES=" << t.peer_summary_bytes << " PINNED_BYTES=" << t.pinned_summary_bytes << '\n';
    print_parameters("AFTER_MU_SIGMA", result.after_mu_sigma);
    print_parameters("AFTER_GAMMA", result.after_gamma);
    print_parameters("AFTER_DELTA_THETA", result.after_delta_theta.parameters);
    std::cout << "PILOT_STATUS=PASS rounds_executed=1\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "PILOT_STATUS=ABORT reason=" << error.what() << '\n';
    return 2;
  }
}
