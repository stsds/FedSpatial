#include "fedspatial/gpu_context.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fedspatial;

struct Expected { int round{}; std::string stage; std::vector<double> value; };
struct Errors { double gamma{}, mu{}, sigma{}, delta{}, alpha{}, range{}; double gamma_rel{}, mu_rel{}, sigma_rel{}, delta_rel{}, alpha_rel{}, range_rel{}; };

static std::vector<Expected> expected_checkpoints() {
  std::ifstream input("sync_gpu/tests/fixtures/sync_gpu/TWO_WORKER_P_CHECKPOINTS.csv");
  if (!input) throw std::runtime_error("missing TWO_WORKER_P_CHECKPOINTS.csv; run export_python_fixture.py");
  std::string line; std::getline(input, line); std::vector<Expected> rows;
  while (std::getline(input, line)) {
    std::stringstream stream(line); std::string field; Expected row;
    std::getline(stream, field, ','); row.round = std::stoi(field);
    std::getline(stream, row.stage, ',');
    while (std::getline(stream, field, ',')) row.value.push_back(std::stod(field));
    if (row.value.size() != 12) throw std::runtime_error("bad checkpoint row");
    rows.push_back(std::move(row));
  }
  return rows;
}

static void observe(Errors* errors, double got, double expected, double* absolute, double* relative) {
  *absolute = std::max(*absolute, std::abs(got - expected));
  // A unit floor makes relative error meaningful for checkpoints at/near zero.
  *relative = std::max(*relative, std::abs(got - expected) / std::max(std::abs(expected), 1.));
}
static void compare(const GlobalParameters& state, const Expected& expected, Errors* errors) {
  observe(errors, state.gamma(0, 0), expected.value[0], &errors->gamma, &errors->gamma_rel);
  observe(errors, state.mu(0, 0), expected.value[1], &errors->mu, &errors->mu_rel);
  observe(errors, state.mu(1, 0), expected.value[2], &errors->mu, &errors->mu_rel);
  observe(errors, state.sigma(0, 0), expected.value[3], &errors->sigma, &errors->sigma_rel);
  observe(errors, state.sigma(0, 1), expected.value[4], &errors->sigma, &errors->sigma_rel);
  observe(errors, state.sigma(1, 0), expected.value[5], &errors->sigma, &errors->sigma_rel);
  observe(errors, state.sigma(1, 1), expected.value[6], &errors->sigma, &errors->sigma_rel);
  observe(errors, state.delta, expected.value[7], &errors->delta, &errors->delta_rel);
  observe(errors, state.alpha, expected.value[8], &errors->alpha, &errors->alpha_rel);
  observe(errors, state.range, expected.value[9], &errors->range, &errors->range_rel);
}

static std::array<WorkerInput, 2> workers() {
  Matrix s0(3, 2), s1(3, 2), z0(3, 1), z1(3, 1), x(3, 1);
  const double sites0[] = {0, 0, 1, .2, .3, 1.1};
  const double sites1[] = {1.2, .9, 1.8, .1, .4, 1.7};
  const double values0[] = {1.2, -.4, .7}, values1[] = {.1, 1.3, -.8};
  std::copy(sites0, sites0 + 6, s0.a.begin()); std::copy(sites1, sites1 + 6, s1.a.begin());
  std::copy(values0, values0 + 3, z0.a.begin()); std::copy(values1, values1 + 3, z1.a.begin());
  for (double& value : x.a) value = 1.;
  return {WorkerInput{s0, z0, x}, WorkerInput{s1, z1, x}};
}
static Matrix knots() { Matrix result(2, 2); const double value[] = {0, 0, 1.5, 1.5}; std::copy(value, value + 4, result.a.begin()); return result; }
static GlobalParameters initial() { Matrix gamma(1, 1), mu(2, 1), sigma(2, 2); gamma(0, 0) = .15; mu(0, 0) = .2; mu(1, 0) = -.1; sigma(0, 0) = .7; sigma(0, 1) = sigma(1, 0) = .05; sigma(1, 1) = .4; return {gamma, mu, sigma, 1.7, 1.2, .8, 3}; }
static std::vector<PTwoWorkerRoundCheckpoints> run(int device0, int device1) {
  TwoWorkerPSynchronousExecutor executor(device0, device1);
  auto current = initial(); const auto input = workers(); const auto knot_matrix = knots();
  std::vector<PTwoWorkerRoundCheckpoints> output;
  for (int round = 0; round < 2; ++round) { auto value = executor.run_round_with_checkpoints(input, knot_matrix, current); current = value.after_delta_theta.parameters; output.push_back(std::move(value)); }
  return output;
}
static bool identical(const std::vector<PTwoWorkerRoundCheckpoints>& left, const std::vector<PTwoWorkerRoundCheckpoints>& right) {
  for (int round = 0; round < 2; ++round) for (const auto* state : {&left[round].after_mu_sigma, &left[round].after_gamma, &left[round].after_delta_theta.parameters}) {
    const GlobalParameters* other = state == &left[round].after_mu_sigma ? &right[round].after_mu_sigma : state == &left[round].after_gamma ? &right[round].after_gamma : &right[round].after_delta_theta.parameters;
    if (state->gamma.a != other->gamma.a || state->mu.a != other->mu.a || state->sigma.a != other->sigma.a ||
        std::memcmp(&state->delta, &other->delta, sizeof(double)) || std::memcmp(&state->alpha, &other->alpha, sizeof(double)) || std::memcmp(&state->range, &other->range, sizeof(double))) return false;
  }
  return true;
}
int main(int argc, char** argv) {
  try {
    int device0 = argc > 1 ? std::stoi(argv[1]) : 1;
    int device1 = argc > 2 ? std::stoi(argv[2]) : device0;
    int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || device0 < 0 || device0 >= count || device1 < 0 || device1 >= count) return 77;
    const auto expected = expected_checkpoints(); if (expected.size() != 6) throw std::runtime_error("expected six Python checkpoints");
    const auto first = run(device0, device1); const auto second = run(device0, device1); Errors errors{}; int index = 0;
    for (int round = 0; round < 2; ++round) {
      for (const char* stage : {"mu_sigma", "gamma", "delta_theta"}) {
        if (expected[index].round != round || expected[index].stage != stage)
          throw std::runtime_error("Python checkpoint ordering mismatch");
        const auto& state = std::string(stage) == "mu_sigma" ? first[round].after_mu_sigma : std::string(stage) == "gamma" ? first[round].after_gamma : first[round].after_delta_theta.parameters;
        compare(state, expected[index], &errors);
        if (std::string(stage) == "delta_theta" &&
            (std::abs(first[round].after_delta_theta.accepted_step - expected[index].value[10]) > 1.e-15 ||
             first[round].after_delta_theta.halving_count != static_cast<int>(expected[index].value[11])))
          throw std::runtime_error("accepted Newton step/halving mismatch");
        ++index;
      }
    }
    const bool deterministic = identical(first, second);
    long long peer_bytes = 0, pinned_bytes = 0; bool raw_stationary = true;
    for (const auto& round : first) { peer_bytes += round.timing.peer_summary_bytes; pinned_bytes += round.timing.pinned_summary_bytes; raw_stationary = raw_stationary && round.timing.raw_data_stationary; }
    // Exercise the actual ordered reduction helper with a non-associative
    // three-worker diagnostic, while production remains exactly two workers.
    Matrix a(1,1), b(1,1), c(1,1); a(0,0)=1.e16; b(0,0)=-1.e16; c(0,0)=1.;
    const double ordered = deterministic_worker_order_mean({a,b,c})(0,0);
    const double reordered = deterministic_worker_order_mean({a,c,b})(0,0);
    const bool order_sensitive = ordered != reordered && ordered == 1./3.;
    const double maximum = std::max({errors.gamma, errors.mu, errors.sigma, errors.delta, errors.alpha, errors.range});
    std::cout << std::setprecision(17)
              << "TWO_WORKER_P_GAMMA_MAX_ABS=" << errors.gamma << " MAX_REL=" << errors.gamma_rel << '\n'
              << "TWO_WORKER_P_MU_MAX_ABS=" << errors.mu << " MAX_REL=" << errors.mu_rel << '\n'
              << "TWO_WORKER_P_SIGMA_MAX_ABS=" << errors.sigma << " MAX_REL=" << errors.sigma_rel << '\n'
              << "TWO_WORKER_P_DELTA_MAX_ABS=" << errors.delta << " MAX_REL=" << errors.delta_rel << '\n'
              << "TWO_WORKER_P_ALPHA_MAX_ABS=" << errors.alpha << " MAX_REL=" << errors.alpha_rel << '\n'
              << "TWO_WORKER_P_RANGE_MAX_ABS=" << errors.range << " MAX_REL=" << errors.range_rel << '\n'
              << "TWO_WORKER_P_DETERMINISM=" << (deterministic ? "BYTE_IDENTICAL" : "FAIL") << '\n'
              << "TWO_WORKER_P2P_SUMMARY_BYTES=" << peer_bytes << '\n'
              << "TWO_WORKER_PINNED_FALLBACK_SUMMARY_BYTES=" << pinned_bytes << '\n'
              << "TWO_WORKER_RAW_DATA_STATIONARY=" << (raw_stationary ? "PASS" : "FAIL") << '\n'
              << "WORKER_ID_ORDER_SENSITIVITY_TEST=" << (order_sensitive ? "PASS" : "FAIL") << '\n'
              << "TWO_WORKER_P_PARITY=" << (maximum <= 1.e-10 && deterministic && raw_stationary && order_sensitive ? "PASS" : "FAIL") << '\n';
    return maximum <= 1.e-10 && deterministic && raw_stationary && order_sensitive ? 0 : 1;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
