#include <cuda_runtime.h>
#include <iostream>

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return 2;
  for (int i = 0; i < count; ++i) for (int j = 0; j < count; ++j) {
    int accessible = 0;
    const cudaError_t status = cudaDeviceCanAccessPeer(&accessible, i, j);
    std::cout << "P2P " << i << ' ' << j << ' ' << (status == cudaSuccess ? accessible : -1) << '\n';
  }
  return 0;
}
