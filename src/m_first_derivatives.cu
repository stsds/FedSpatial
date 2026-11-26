#include "fedspatial/gpu_context.hpp"
namespace fedspatial { namespace {
Matrix gpu_scale(GpuContext&c,const Matrix&x,double s){Matrix out(x.rows,x.cols);double*d;cudaMalloc(&d,x.a.size()*sizeof(double));cudaMemcpyAsync(d,x.a.data(),x.a.size()*sizeof(double),cudaMemcpyHostToDevice,c.stream());cublasDscal(c.cublas(),x.a.size(),&s,d,1);cudaMemcpyAsync(out.a.data(),d,x.a.size()*sizeof(double),cudaMemcpyDeviceToHost,c.stream());cudaStreamSynchronize(c.stream());cudaFree(d);return out;}
}
MFirstDerivatives compute_m_first_derivatives(GpuContext&c,const WorkerInput&w,const Matrix&knots,const GlobalParameters&p){PProfileScope profile_scope(c.profile(), "compute_m_first_derivatives");c.bind_region(w,knots);return download_device_m_derivatives(c,compute_device_m_derivatives(c,p.nu2,p.delta,p.alpha,p.range)).first;}
MSecondDerivatives compute_m_second_derivatives(GpuContext&c,const WorkerInput&w,const Matrix&knots,const GlobalParameters&p){PProfileScope profile_scope(c.profile(), "compute_m_second_derivatives");c.bind_region(w,knots);return download_device_m_derivatives(c,compute_device_m_derivatives(c,p.nu2,p.delta,p.alpha,p.range)).second;}
MSecondDerivatives assemble_m_second_derivatives(GpuContext&c,const MFirstDerivatives&first,const ASecondDerivatives&a,double delta){PProfileScope profile_scope(c.profile(), "compute_m_second_derivatives");Matrix zero=gpu_scale(c,first.delta,0),da=gpu_scale(c,first.alpha,1/delta),dr=gpu_scale(c,first.range,1/delta);return {zero,da,dr,da,gpu_scale(c,a.aa,delta),gpu_scale(c,a.ar,delta),dr,gpu_scale(c,a.ra,delta),gpu_scale(c,a.rr,delta)};}
}
