#include "fedspatial/gpu_context.hpp"
#include <chrono>
#include <iostream>

using namespace fedspatial;
int main() {
  Matrix sites(3,2), z(3,1), x(3,1), knots(2,2);
  const double sv[]={0.,0.,1.,.2,.3,1.1}, kv[]={0.,0.,1.5,1.5};
  for(int i=0;i<6;++i) sites.a[i]=sv[i]; for(int i=0;i<4;++i) knots.a[i]=kv[i]; for(double& v:x.a) v=1.;
  WorkerInput worker{sites,z,x};
  try { GpuContext context(0); context.bind_region(worker,knots); const auto before=context.raw_h2d_bytes();
    const auto result_before=context.result_d2h_bytes();
    const auto device_start=std::chrono::steady_clock::now();
    const auto slice=compute_device_b_slice(context,3,1.2,.8);
    cudaStreamSynchronize(context.stream());
    const auto device_end=std::chrono::steady_clock::now();
    const auto after_device=context.raw_h2d_bytes(); const auto result_after_device=context.result_d2h_bytes();
    const Matrix gpu=download_device_b_slice(context,slice); const auto after_download=context.raw_h2d_bytes(); const auto result_after_download=context.result_d2h_bytes();
    const Matrix c=matern15(sites,knots,1.2,.8), k=matern15(knots,knots,1.2,.8);
    const Matrix cpu=transpose(solve_spd(k,transpose(c))); const double error=max_abs_diff(cpu,gpu);
    const auto ms=std::chrono::duration<double,std::milli>(device_end-device_start).count();
    const bool pass=error<=1e-12&&before==after_device&&after_device==after_download&&result_before==result_after_device;
    std::cout<<"B_SLICE_MAX_ABS="<<error<<" MAX_REL="<<error/1.<<" DEVICE_MS="<<ms
             <<"\nRAW_H2D_INITIAL="<<before<<" RAW_H2D_AFTER_DEVICE="<<after_device<<" RAW_H2D_AFTER_DOWNLOAD="<<after_download
             <<"\nRESULT_D2H_BEFORE="<<result_before<<" RESULT_D2H_AFTER_DEVICE="<<result_after_device<<" RESULT_D2H_AFTER_TEST_DOWNLOAD="<<result_after_download
             <<'\n'<<"DEVICE_B_SLICE_PARITY="<<(pass?"PASS":"FAIL")<<'\n';
    return pass?0:1;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 2; }
}
