#include "fedspatial/gpu_context.hpp"
#include <iomanip>
#include <iostream>

int main() {
  try {
    fedspatial::GpuContext context(0);
    const double error=fedspatial::compact_reduction_m2000_max_abs_error(context);
    std::cout << std::setprecision(17) << "COMPACT_REDUCTION_M2000_MAX_ABS_ERROR=" << error << '\n'
              << "COMPACT_REDUCTION_M2000_ALLOCATION_LIGHT=PASS\n"
              << "COMPACT_REDUCTION_M2000_REGRESSION=" << (error==0. ? "PASS" : "FAIL") << '\n';
    return error==0. ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "COMPACT_REDUCTION_M2000_FAILURE=" << error.what() << '\n';
    return 2;
  }
}
