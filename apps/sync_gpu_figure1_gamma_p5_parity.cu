#include "fedspatial/gpu_context.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace fedspatial;
namespace fs=std::filesystem;
static Matrix load(const fs::path&p){Matrix a;if(!read_matrix(p.string(),&a))throw std::runtime_error("missing "+p.string());return a;}
static void add(Matrix&a,const Matrix&b){for(int i=0;i<a.rows*a.cols;i++)a.a[i]+=b.a[i];}
static void scale(Matrix&a,double s){for(double&v:a.a)v*=s;}
static void rows(std::ofstream&o,const char*q,int worker,const Matrix&py,const Matrix&cu,double&mx){for(int i=0;i<py.rows;i++)for(int j=0;j<py.cols;j++){double e=std::abs(py(i,j)-cu(i,j));mx=std::max(mx,e);o<<worker<<','<<q<<','<<i<<','<<j<<','<<std::setprecision(17)<<py(i,j)<<','<<cu(i,j)<<','<<e<<','<<e/std::max(1e-300,std::abs(py(i,j)))<<'\n';}}
int main(){try{const fs::path f="sync_gpu/tests/fixtures/figure1_reference",out="results/figure1_cpp_gpu_faithful/parity";fs::create_directories(out);Matrix knots=load(f/"knots.txt"),mu=load(f/"p_initial_global_mu.txt"),sigma=load(f/"p_initial_global_sigma.txt"),gamma(5,1);double gv[]={-1,2,3,-2,1};for(int i=0;i<5;i++)gamma(i,0)=gv[i];GlobalParameters p{gamma,mu,sigma,.25,1.,.11};GpuContext c(0);Matrix ag(5,1),ax(5,5);std::ofstream csv(out/"gamma_p5_parity.csv");csv<<"worker_id,quantity,row,column,python_value,cpp_value,abs_error,rel_error\n";double eg=0,ex=0,eglobal=0,enew=0;for(int w=0;w<10;w++){Matrix d=load(f/("worker_"+std::to_string(w)+".txt"));if(d.rows!=100||d.cols!=8)throw std::runtime_error("unexpected Figure 1 worker shape");WorkerInput in{Matrix(100,2),Matrix(100,1),Matrix(100,5)};for(int i=0;i<100;i++){in.sites(i,0)=d(i,0);in.sites(i,1)=d(i,1);in.z(i,0)=d(i,2);for(int k=0;k<5;k++)in.x(i,k)=d(i,3+k);}auto got=compute_gamma_local(c,in,knots,p);Matrix pyg=load(f/("p_initial_worker_"+std::to_string(w)+"_gamma.txt")),pyx=load(f/("p_initial_worker_"+std::to_string(w)+"_xx.txt"));rows(csv,"gamma_local",w,pyg,got.gamma,eg);rows(csv,"XX_local",w,pyx,got.xx,ex);add(ag,got.gamma);add(ax,got.xx);}scale(ag,.1);scale(ax,.1);Matrix pyg=load(f/"p_initial_global_gamma.txt"),pyx=load(f/"p_initial_global_xx.txt"),pynew=load(f/"p_initial_gamma_new.txt"),newg=solve_spd(ax,ag);rows(csv,"gamma_global",-1,pyg,ag,eglobal);rows(csv,"XX_global",-1,pyx,ax,eglobal);rows(csv,"gamma_new",-1,pynew,newg,enew);std::cout<<"P5_GAMMA_LOCAL_MAX_ABS_ERROR="<<eg<<"\nP5_XX_LOCAL_MAX_ABS_ERROR="<<ex<<"\nP5_GAMMA_GLOBAL_MAX_ABS_ERROR="<<eglobal<<"\nP5_GAMMA_NEW_MAX_ABS_ERROR="<<enew<<"\nP5_GAMMA_PARITY="<<(std::max({eg,ex,eglobal,enew})<=1e-10?"PASS":"FAIL")<<"\n";return std::max({eg,ex,eglobal,enew})<=1e-10?0:1;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
