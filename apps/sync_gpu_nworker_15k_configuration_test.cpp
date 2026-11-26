#include "fedspatial/nworker_15k_figure1_config.hpp"

#include <iostream>

using namespace fedspatial;

int main() {
  const auto p = nworker_15k_figure1_initial();
  print_nworker_15k_figure1_config(std::cout);
  const bool pass = p.gamma.rows == 5 && p.gamma(0, 0) == -1. &&
      p.gamma(1, 0) == 2. && p.gamma(2, 0) == 3. && p.gamma(3, 0) == -2. &&
      p.gamma(4, 0) == 1. && p.delta == .25 && p.alpha == 1. &&
      p.range == .11 && p.nu2 == 3 && NWorker15kFigure1Config::workers == 2 &&
      NWorker15kFigure1Config::total_n == 30000 &&
      NWorker15kFigure1Config::cuda_devices[0] == 1 &&
      NWorker15kFigure1Config::cuda_devices[1] == 2;
  std::cout << "FIGURE1_CONFIGURATION_ONLY=" << (pass ? "PASS" : "FAIL") << '\n';
  return pass ? 0 : 1;
}
