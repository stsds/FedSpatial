#include "fedspatial/gpu_context.hpp"
#include <filesystem>
#include <iostream>
using namespace fedspatial; namespace fs=std::filesystem;
int main(){fs::path f="sync_gpu/tests/fixtures/figure1_reference";Matrix d,k;if(!read_matrix((f/"worker_0.txt").string(),&d)||!read_matrix((f/"knots.txt").string(),&k))return 2;Matrix s(100,2);for(int i=0;i<100;i++){s(i,0)=d(i,0);s(i,1)=d(i,1);}GpuContext c(0);double worst=0;for(int nu:{1,3,5}){auto old=matern_with_derivatives(c,s,k,nu,1.,.11);auto now=compute_cnk_derivatives_fused(c,s,k,nu,1.,.11);const Matrix* a[]={&old.alpha,&old.range,&old.aa,&old.ar,&old.rr};const Matrix* b[]={&now.alpha,&now.range,&now.alpha_alpha,&now.alpha_range,&now.range_range};const char*n[]={"alpha","range","alpha_alpha","alpha_range","range_range"};for(int q=0;q<5;q++){double e=max_abs_diff(*a[q],*b[q]);worst=std::max(worst,e);std::cout<<"NU="<<nu<<" "<<n[q]<<" DIM="<<a[q]->rows<<"x"<<a[q]->cols<<" MAX_ABS="<<e<<"\n";}}return worst<1e-12?0:1;}
