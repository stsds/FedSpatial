#include "fedspatial/figure1_fixture.hpp"
#include "fedspatial/gpu_context.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fedspatial;
namespace fs = std::filesystem;

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

GlobalParameters initial(int q, int m) {
  if (q != 5) throw std::runtime_error("benchmark expects five covariates");
  Matrix gamma(q,1), mu(m,1), sigma(m,m);
  const double values[5] = {-1.,2.,3.,-2.,1.};
  for (int i=0;i<q;++i) gamma(i,0)=values[i];
  return {gamma,mu,sigma,.25,1.,.11,3};
}

double block1_flops(int n,int m) {
  // Algorithmic FP64 work in the dense local Block-1 path: covariance/basis
  // products, dense Cholesky, triangular solves, and summary contractions.
  return 6.0*n*m*m + 4.0*n*n*m + (double(n)*n*n)/3.0 +
         2.0*n*n + 2.0*n*m;
}
}

int main(int argc,char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: block1_flops FIXTURE REPEATS");
    const fs::path fixture_path=argv[1];
    const int repeats=std::stoi(argv[2]);
    if (repeats<1) throw std::runtime_error("repeats must be positive");
    Figure1Fixture fixture;
    if (!load_figure1_python_fixture(fixture_path,&fixture))
      throw std::runtime_error("fixture load failed");
    const auto worker=partition_figure1_fixture(fixture,1).at(0);
    const int n=worker.sites.rows,m=fixture.knots.rows;
    auto parameters=initial(fixture.x.cols,m);
    GpuContext context(0);
    context.bind_region(worker,fixture.knots);
    (void)build_device_mu_sigma_local(context,worker,fixture.knots,parameters);
    check(cudaStreamSynchronize(context.stream()));

    cudaEvent_t start{},stop{};
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    std::cout << "n,m,repetition,local_block1_seconds,algorithmic_gflops\n";
    for (int repetition=0;repetition<repeats;++repetition) {
      // A gamma change invalidates the parameter state without changing the
      // covariance problem or its mathematical workload.
      parameters.gamma(0,0)+=1e-14;
      check(cudaEventRecord(start,context.stream()));
      (void)build_device_mu_sigma_local(context,worker,fixture.knots,parameters);
      check(cudaEventRecord(stop,context.stream()));
      check(cudaEventSynchronize(stop));
      float milliseconds=0.f; check(cudaEventElapsedTime(&milliseconds,start,stop));
      const double seconds=milliseconds/1000.0;
      std::cout << n << ',' << m << ',' << repetition << ','
                << std::setprecision(10) << seconds << ','
                << block1_flops(n,m)/seconds/1e9 << '\n';
    }
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    std::cout << "BLOCK1_FLOPS_STATUS=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "BLOCK1_FLOPS_FAILURE=" << error.what() << '\n';
    return 2;
  }
}
