#include "fedspatial/gpu_context.hpp"
namespace fedspatial {
ASecondDerivatives compute_a_second_derivatives(GpuContext&c,const WorkerInput&w,const Matrix&knots,const GlobalParameters&p){PProfileScope profile_scope(c.profile(), "compute_a_second_derivatives");c.bind_region(w,knots);return download_device_a_second_derivatives(c,compute_device_a_derivatives(c,p.nu2,p.alpha,p.range));}
ASecondDerivatives compute_a_second_derivatives_from_b(GpuContext&c,const WorkerInput&w,const Matrix&knots,const GlobalParameters&p,const BSecondDerivatives&){PProfileScope profile_scope(c.profile(), "compute_a_second_derivatives");c.bind_region(w,knots);return download_device_a_second_derivatives(c,compute_device_a_derivatives(c,p.nu2,p.alpha,p.range));}
}
