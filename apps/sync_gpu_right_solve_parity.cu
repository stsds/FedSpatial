#include "fedspatial/gpu_context.hpp"
#include <filesystem>
#include <iostream>
using namespace fedspatial; namespace fs=std::filesystem;
int main(){fs::path f="sync_gpu/tests/fixtures/figure1_reference";Matrix k,s;if(!read_matrix((f/"knots.txt").string(),&k)||!read_matrix((f/"worker_0.txt").string(),&s))return 2;Matrix x(100,100);for(int i=0;i<100;i++)for(int j=0;j<100;j++)x(i,j)=s(i,j%8+2);Matrix K=matern15(k,k,1.,.11),cpu=transpose(solve_spd(K,transpose(x))),L=cholesky_lower(K);GpuContext c(0);Matrix gpu=gpu_right_spd_solve(c,x,L),repeat=gpu_right_spd_solve(c,x,L),res=subtract(matmul(gpu,K),x);double error=std::max(max_abs_diff(cpu,gpu),max_abs_diff(gpu,repeat));std::cout<<"MAX_ABS="<<error<<" RESIDUAL="<<max_abs_diff(res,Matrix(100,100))<<" ALLOCS="<<c.right_solve_allocation_count()<<" FACTOR_UPLOADS="<<c.right_solve_factor_upload_count()<<"\n";return error<1e-12&&c.right_solve_allocation_count()==2&&c.right_solve_factor_upload_count()==1?0:1;}
