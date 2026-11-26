#include "fedspatial/gpu_context.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
using namespace fedspatial;

static Matrix load(const char* p){std::ifstream f(p);int r,c;if(!(f>>r>>c))throw std::runtime_error("missing fixture");Matrix x(r,c);for(double&v:x.a)f>>v;return x;}
static WorkerInput fixture(int n){auto a=load("sync_gpu/tests/fixtures/figure1_reference/worker_0.txt");Matrix s(n,2),z(n,1),x(n,5);for(int i=0;i<n;++i){s(i,0)=a(i,0);s(i,1)=a(i,1);z(i,0)=a(i,2);for(int j=0;j<5;++j)x(i,j)=a(i,j+3);}return{s,z,x};}
static Matrix reference(GpuContext& c,const WorkerInput&w,const Matrix&k,int parameter){
  constexpr double alpha=1.,range=.11; const auto cn=matern_with_derivatives(c,w.sites,k,3,alpha,range);const auto kk=matern_with_derivatives(c,k,k,3,alpha,range);
  const Matrix B=transpose(solve_spd(kk.value,transpose(cn.value)));
  const Matrix ci=parameter?cn.range:cn.alpha, ki=parameter?kk.range:kk.alpha;
  Matrix answer=transpose(solve_spd(kk.value,transpose(ci)));
  const Matrix correction=transpose(solve_spd(kk.value,transpose(matmul(B,ki))));
  for(size_t i=0;i<answer.a.size();++i)answer.a[i]-=correction.a[i]; return answer;
}
static bool run(int n){
  auto knots=load("sync_gpu/tests/fixtures/figure1_reference/knots.txt"); auto w=fixture(n); GpuContext c(0);c.bind_region(w,knots);
  const auto before_raw=c.raw_h2d_bytes(),before_d2h=c.result_d2h_bytes(),combined_before=c.b_derivative_allocation_count(),first_before=c.b_first_derivative_allocation_count();
  double ea=0.,er=0.,ra=0.,rr=0.,repeat=0.;
  long long active_d2h=0;
  for(int p=0;p<2;++p){const Matrix want=reference(c,w,knots,p);const auto d2h_before_active=c.result_d2h_bytes();auto s=compute_device_b_first_derivative(c,3,1.,.11,p);active_d2h+=c.result_d2h_bytes()-d2h_before_active;Matrix got(n,knots.rows);cudaMemcpyAsync(got.a.data(),s.derivative,got.a.size()*sizeof(double),cudaMemcpyDeviceToHost,c.stream());cudaStreamSynchronize(c.stream());c.record_model_d2h_bytes(got.a.size()*sizeof(double));const auto d2h_before_repeat=c.result_d2h_bytes();auto again=compute_device_b_first_derivative(c,3,1.,.11,p);active_d2h+=c.result_d2h_bytes()-d2h_before_repeat;Matrix twice(n,knots.rows);cudaMemcpyAsync(twice.a.data(),again.derivative,twice.a.size()*sizeof(double),cudaMemcpyDeviceToHost,c.stream());cudaStreamSynchronize(c.stream());c.record_model_d2h_bytes(twice.a.size()*sizeof(double));double scale=1.;for(double v:want.a)scale=std::max(scale,std::abs(v));const double e=max_abs_diff(want,got), rel=e/scale;repeat=std::max(repeat,max_abs_diff(got,twice));if(p==0){ea=e;ra=rel;}else{er=e;rr=rel;}}
  const long long diagnostic_d2h=4LL*n*knots.rows*sizeof(double);
  const bool transfer=c.raw_h2d_bytes()==before_raw && active_d2h==0 && c.result_d2h_bytes()==before_d2h+diagnostic_d2h;
  const bool allocation=c.b_derivative_allocation_count()==combined_before && c.b_first_derivative_allocation_count()==first_before+1;
  const bool ok=std::max({ea,er,repeat})<=1e-10&&std::max(ra,rr)<=1e-10&&transfer&&allocation;
  std::cout<<"B_FIRST_N="<<n<<" ALPHA_MAX_ABS="<<ea<<" RANGE_MAX_ABS="<<er<<" MAX_REL="<<std::max(ra,rr)<<" EXACT_REPEAT_ABS="<<repeat<<"\nB_FIRST_ACTIVE_H2D_BYTES="<<c.raw_h2d_bytes()-before_raw<<" B_FIRST_ACTIVE_D2H_BYTES="<<active_d2h<<" B_FIRST_TEST_D2H_BYTES="<<diagnostic_d2h<<" COMBINED_B_ALLOCATIONS="<<c.b_derivative_allocation_count()-combined_before<<" FIRST_B_ALLOCATIONS="<<c.b_first_derivative_allocation_count()-first_before<<"\n";return ok;
}
int main(){try{const bool ok=run(3)&&run(100);std::cout<<"DEVICE_B_FIRST_DERIVATIVE_PARITY="<<(ok?"PASS":"FAIL")<<'\n';return ok?0:1;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
