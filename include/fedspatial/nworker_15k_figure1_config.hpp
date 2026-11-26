#pragma once

#include "fedspatial/gpu_context.hpp"

#include <array>
#include <ostream>

namespace fedspatial {

// The 2x15k case is a scale-up of the checked Figure-1 scenario-4 fixture,
// not an independent synthetic experiment.  It is generated as one global
// 30k-row realization and then partitioned into consecutive worker blocks.
struct NWorker15kFigure1Config {
  // Dedicated full-scale Figure-1 P target.  This is intentionally distinct
  // from the generic N-worker executor and its regression matrix.
  static constexpr int workers = 2;
  static constexpr int local_n = 15000;
  static constexpr int total_n = workers * local_n;
  static constexpr int knots = 2000;
  static constexpr int covariates = 5;
  static constexpr int rounds = 50;
  static constexpr int scenario_id = 4;
  static constexpr int replication = 0;
  static constexpr std::uint64_t torch_seed = 1;
  static constexpr int nu2 = 3;  // 2 * nu, so nu = 1.5.
  static constexpr double delta = .25;
  static constexpr double alpha = 1.;
  static constexpr double beta = .11;
  static constexpr std::array<int, workers> cuda_devices{{1, 2}};
  static constexpr const char* gv100_uuid_0 = "GPU-737f3c7a-bf62-20b7-01f8-b3fb1a3c86f4";
  static constexpr const char* gv100_uuid_1 = "GPU-a7c40d19-eb27-1d13-257a-5949f98fc947";
  static constexpr const char* gv100_bus_0 = "0000:AF:00.0";
  static constexpr const char* gv100_bus_1 = "0000:D8:00.0";
  static constexpr std::array<double, covariates> gamma{{-1., 2., 3., -2., 1.}};
};

// Deliberately separate timing preset.  It shares the Figure-1 generator
// contract and physical-device policy, but never changes the 2x15k target.
struct NWorker2kFigure1TimingConfig {
  static constexpr int workers = 2;
  static constexpr int local_n = 2000;
  static constexpr int total_n = workers * local_n;
  static constexpr int knots = 2000;
  static constexpr int covariates = 5;
  static constexpr int rounds = 1;
  static constexpr int scenario_id = 4, replication = 0;
  static constexpr std::uint64_t torch_seed = 1;
  static constexpr int nu2 = 3;
  static constexpr double delta = .25, alpha = 1., beta = .11;
  static constexpr std::array<int, workers> cuda_devices{{1,2}};
  static constexpr const char* gv100_uuid_0 = NWorker15kFigure1Config::gv100_uuid_0;
  static constexpr const char* gv100_uuid_1 = NWorker15kFigure1Config::gv100_uuid_1;
  static constexpr const char* gv100_bus_0 = NWorker15kFigure1Config::gv100_bus_0;
  static constexpr const char* gv100_bus_1 = NWorker15kFigure1Config::gv100_bus_1;
};

// Canonical two-GV100 optimization target.  It is distinct from the m=2000
// timing preset so neither target can silently accept the other's fixture.
struct NWorker2kM400Figure1Config {
  static constexpr int workers = 2;
  static constexpr int local_n = 2000;
  static constexpr int total_n = workers * local_n;
  static constexpr int knots = 400;
  static constexpr int covariates = 5;
  static constexpr int rounds = 1;
  static constexpr int scenario_id = 4, replication = 0;
  static constexpr std::uint64_t torch_seed = 1;
  static constexpr int nu2 = 3;
  static constexpr double delta = .25, alpha = 1., beta = .11;
  static constexpr std::array<int, workers> cuda_devices{{1,2}};
  static constexpr const char* gv100_uuid_0 = NWorker15kFigure1Config::gv100_uuid_0;
  static constexpr const char* gv100_uuid_1 = NWorker15kFigure1Config::gv100_uuid_1;
  static constexpr const char* gv100_bus_0 = NWorker15kFigure1Config::gv100_bus_0;
  static constexpr const char* gv100_bus_1 = NWorker15kFigure1Config::gv100_bus_1;
};

// Full-replica smoke preset for the canonical optimization target.  It is
// intentionally separate from the one-round timing preset so a smoke run
// cannot silently consume a different round count or fixture shape.
struct NWorker2kM400Figure1SmokeConfig : NWorker2kM400Figure1Config {
  static constexpr int rounds = 50;
};

inline GlobalParameters nworker_15k_figure1_initial(int m = NWorker15kFigure1Config::knots) {
  Matrix gamma(NWorker15kFigure1Config::covariates, 1), mu(m, 1), sigma(m, m);
  for (int i = 0; i < NWorker15kFigure1Config::covariates; ++i)
    gamma(i, 0) = NWorker15kFigure1Config::gamma[i];
  for (int i = 0; i < m; ++i) sigma(i, i) = 1.;
  return {gamma, mu, sigma, NWorker15kFigure1Config::delta,
          NWorker15kFigure1Config::alpha, NWorker15kFigure1Config::beta,
          NWorker15kFigure1Config::nu2};
}

inline void print_nworker_15k_figure1_config(std::ostream& out) {
  out << "FIGURE1_CONFIG scenario_id=" << NWorker15kFigure1Config::scenario_id
      << " replication=" << NWorker15kFigure1Config::replication
      << " torch_seed=" << NWorker15kFigure1Config::torch_seed
      << " generator=GPPSampleGeneratorUnitSquare.generate_obs_gp"
      << " coordinate_order=global_jittered_grid_then_consecutive_torch_chunk"
      << " worker_reduction_order=worker_id_ascending"
      << " workers=" << NWorker15kFigure1Config::workers
      << " cuda_devices=1,2 device0_forbidden=YES"
      << " n_per_worker=" << NWorker15kFigure1Config::local_n
      << " total_n=" << NWorker15kFigure1Config::total_n
      << " m=" << NWorker15kFigure1Config::knots
      << " q=" << NWorker15kFigure1Config::covariates
      << " rounds=" << NWorker15kFigure1Config::rounds
      << " nu=" << (NWorker15kFigure1Config::nu2 / 2.)
      << " nu2=" << NWorker15kFigure1Config::nu2
      << " delta=" << NWorker15kFigure1Config::delta
      << " alpha=" << NWorker15kFigure1Config::alpha
      << " beta=" << NWorker15kFigure1Config::beta
      << " gamma=-1,2,3,-2,1 noise_sd=2" << '\n';
}

}  // namespace fedspatial
