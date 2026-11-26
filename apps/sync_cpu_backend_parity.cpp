#include "fedspatial/cpu_backend.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <cmath>
#include <stdexcept>
#ifdef FEDSPATIAL_SCALAPACK
#include <mpi.h>
#endif
using namespace fedspatial;
namespace fs=std::filesystem;
namespace {
struct Error {double abs=0,rel=0;int checks=0;};
std::map<std::string,Error> errors;
Matrix load(const fs::path& path){Matrix x;if(!read_matrix(path.string(),&x))throw std::runtime_error("missing "+path.string());return x;}
void check(const std::string& block,const Matrix& x,const Matrix& ref,double tolerance=1e-10) {
  if(x.rows!=ref.rows||x.cols!=ref.cols)throw std::runtime_error("shape mismatch "+block);
  auto& e=errors[block];++e.checks;
  for(size_t i=0;i<x.a.size();++i){double d=std::abs(x.a[i]-ref.a[i]);if(!std::isfinite(d))throw std::runtime_error("non-finite "+block);
    e.abs=std::max(e.abs,d);e.rel=std::max(e.rel,d/std::max(1.,std::abs(ref.a[i])));
    if(d>tolerance)throw std::runtime_error(block+" exceeds tolerance: "+std::to_string(d));}
}
Matrix scalar(double v){Matrix x(1,1);x(0,0)=v;return x;}
Matrix vector(const std::array<double,3>& v){Matrix x(3,1);x.a.assign(v.begin(),v.end());return x;}
Matrix hessian(const std::array<double,9>& v){Matrix x(3,3);x.a.assign(v.begin(),v.end());return x;}
WorkerInput worker(const fs::path& path){auto d=load(path);WorkerInput w{Matrix(d.rows,2),Matrix(d.rows,1),Matrix(d.rows,d.cols-3)};
  for(int i=0;i<d.rows;++i){w.sites(i,0)=d(i,0);w.sites(i,1)=d(i,1);w.z(i,0)=d(i,2);for(int j=0;j<w.x.cols;++j)w.x(i,j)=d(i,j+3);}return w;}
GlobalParameters initial(){GlobalParameters p{Matrix(1,1),Matrix(2,1),Matrix(2,2),1.7,1.2,.8,3};p.gamma(0,0)=.15;p.mu(0,0)=.2;p.mu(1,0)=-.1;p.sigma.a={.7,.05,.05,.4};return p;}
void small(const fs::path& dir) {
  Matrix knots(2,2);knots(1,0)=knots(1,1)=1.5;auto p=initial();
  for(int w=0;w<2;++w){auto input=worker(dir/("N_WORKER_"+std::to_string(w)+".txt"));auto local=cpu_local(input,knots,p);std::string id=std::to_string(w);
    check("K",cpu_matern(knots,knots,3,p.alpha,p.range).value,load(dir/(w?"K1.txt":"K.txt")));
    check("B_reconstruction",matmul(local.B,cpu_matern(knots,knots,3,p.alpha,p.range).value),
          cpu_matern(input.sites,knots,3,p.alpha,p.range).value);
    check("M",local.M,load(dir/("M"+id+".txt")));
    check("M_inverse",inverse_spd(local.M),load(dir/("Minv"+id+".txt")));
    check("Cholesky",matmul(local.chol_M,transpose(local.chol_M)),local.M);
    check("mu_summary",local.summary.mu,load(dir/("TWO_WORKER"+id+"_LOCAL_MU.txt")));
    check("sigma_summary",local.summary.sigma,load(dir/("TWO_WORKER"+id+"_LOCAL_SIGMA.txt")));
    auto gamma=cpu_gamma(input,knots,p);
    check("gamma_summary",gamma.gamma,load(dir/("TWO_WORKER"+id+"_GAMMA_LOCAL_GAMMA.txt")));
    check("XX_summary",gamma.xx,load(dir/("TWO_WORKER"+id+"_GAMMA_LOCAL_XX.txt")));
    check("SPD_solve",matmul(local.M,solve_spd(local.M,input.x)),input.x);
  }
  auto input=worker(dir/"N_WORKER_0.txt");
  for(int point=0;point<7;++point){std::string prefix="DELTA_THETA_POINT"+std::to_string(point)+"_";
    auto v=load(dir/(prefix+"PARAM.txt"));p.delta=v.a[0];p.alpha=v.a[1];p.range=v.a[2];
    auto dt=cpu_delta_theta(input,knots,p);auto d=cpu_derivatives(input,knots,p);
    check("objective",scalar(dt.objective),load(dir/(prefix+"OBJECTIVE.txt")));
    check("gradient",vector(dt.gradient),load(dir/(prefix+"GRAD.txt")));
    check("Hessian",hessian(dt.hessian),load(dir/(prefix+"HESSIAN.txt")));
    for(auto pair:{std::pair{"K",cpu_matern(knots,knots,3,p.alpha,p.range)},
        std::pair{"CROSS",cpu_matern(input.sites,knots,3,p.alpha,p.range)},
        std::pair{"CJJ",cpu_matern(input.sites,input.sites,3,p.alpha,p.range)}}){
      const auto& c=pair.second;const Matrix* values[]={&c.aa,&c.ar,&c.ra,&c.rr};const char* names[]={"AA","AR","RA","RR"};
      for(int i=0;i<4;++i)check(std::string(pair.first)+"_second",*values[i],load(dir/(prefix+pair.first+"_HESS_"+names[i]+".txt")));
    }
    const Matrix* bs[]={&d.b_second.baa,&d.b_second.bar,&d.b_second.bra,&d.b_second.brr};
    const Matrix* as[]={&d.a_second.aa,&d.a_second.ar,&d.a_second.ra,&d.a_second.rr};
    const char* names[]={"AA","AR","RA","RR"};
    for(int i=0;i<4;++i){check("B_second",*bs[i],load(dir/(prefix+"B_HESS_"+names[i]+".txt")));check("A_second",*as[i],load(dir/(prefix+"A_HESS_"+names[i]+".txt")));}
    const Matrix* mf[]={&d.m_first.delta,&d.m_first.alpha,&d.m_first.range};const char* mn[]={"DELTA","ALPHA","RANGE"};
    for(int i=0;i<3;++i)check("M_first",*mf[i],load(dir/(prefix+"M_GRAD_"+mn[i]+".txt")));
    const Matrix* ms[]={&d.m_second.dd,&d.m_second.da,&d.m_second.dr,&d.m_second.ad,&d.m_second.aa,&d.m_second.ar,&d.m_second.rd,&d.m_second.ra,&d.m_second.rr};
    const char* sn[]={"DD","DA","DR","AD","AA","AR","RD","RA","RR"};
    for(int i=0;i<9;++i)check("M_second",*ms[i],load(dir/(prefix+"M_HESS_"+sn[i]+".txt")));
  }
  p=initial();auto local=cpu_local(input,knots,p);auto state=p;cpu_update_mu_sigma(state,knots,local.summary,1);
  check("mu_update",state.mu,load(dir/"MU_SIGMA_OUT_MU.txt"));check("sigma_update",state.sigma,load(dir/"MU_SIGMA_OUT_SIGMA.txt"));
  auto gamma=cpu_gamma(input,knots,p);check("gamma_update",solve_spd(gamma.xx,gamma.gamma),load(dir/"GAMMA_OUT.txt"));
  // Independently evaluate the three supported smoothness primitives.
  for(int nu2:{1,3,5}) {
    auto d=cpu_matern(input.sites,knots,nu2,1.2,.8);Matrix expected(input.sites.rows,knots.rows);
    for(int i=0;i<expected.rows;++i)for(int j=0;j<expected.cols;++j){double dx=input.sites(i,0)-knots(j,0),dy=input.sites(i,1)-knots(j,1),u=std::sqrt(nu2*(dx*dx+dy*dy))/.8;
      expected(i,j)=1.2*std::exp(-u)*(nu2==1?1.:nu2==3?1.+u:1.+u+u*u/3.);}
    check("Matern_nu"+std::to_string(nu2),d.value,expected);
  }
}
void figure1(const fs::path& dir) {
  auto knots=load(dir/"knots.txt");GlobalParameters p{Matrix(5,1),Matrix(knots.rows,1),Matrix(knots.rows,knots.rows),.25,1.,.11,3};p.gamma.a={-1,2,3,-2,1};
  MuSigmaLocalSummary mean{Matrix(knots.rows,1),Matrix(knots.rows,knots.rows)};
  for(int w=0;w<10;++w){auto local=cpu_local(worker(dir/("worker_"+std::to_string(w)+".txt")),knots,p).summary;
    auto prefix="p_initial_worker_"+std::to_string(w)+"_";
    check("p5_mu_summary",local.mu,load(dir/(prefix+"mu.txt")));check("p5_sigma_summary",local.sigma,load(dir/(prefix+"sigma.txt")));
    for(size_t i=0;i<mean.mu.a.size();++i)mean.mu.a[i]+=local.mu.a[i];for(size_t i=0;i<mean.sigma.a.size();++i)mean.sigma.a[i]+=local.sigma.a[i];}
  for(double& x:mean.mu.a)x/=10;for(double& x:mean.sigma.a)x/=10;cpu_update_mu_sigma(p,knots,mean,10);
  check("p5_mu_update",p.mu,load(dir/"p_initial_global_mu.txt"));check("p5_sigma_update",p.sigma,load(dir/"p_initial_global_sigma.txt"));
  Matrix rhs(5,1),xx(5,5);
  for(int w=0;w<10;++w){auto local=cpu_gamma(worker(dir/("worker_"+std::to_string(w)+".txt")),knots,p);auto prefix="p_initial_worker_"+std::to_string(w)+"_";
    check("p5_gamma_summary",local.gamma,load(dir/(prefix+"gamma.txt")));check("p5_XX_summary",local.xx,load(dir/(prefix+"xx.txt")));
    for(int i=0;i<5;++i)rhs.a[i]+=local.gamma.a[i];for(int i=0;i<25;++i)xx.a[i]+=local.xx.a[i];}
  for(double& x:rhs.a)x/=10;for(double& x:xx.a)x/=10;p.gamma=solve_spd(xx,rhs);
  check("p5_gamma_update",p.gamma,load(dir/"p_initial_gamma_new.txt"));
  Matrix g(3,1),h(3,3),kg(3,1),kh(3,3);
  for(int w=0;w<10;++w){auto local=cpu_delta_theta(worker(dir/("worker_"+std::to_string(w)+".txt")),knots,p);auto prefix="p_initial_worker_"+std::to_string(w)+"_";
    check("p5_objective",scalar(local.objective),load(dir/(prefix+"objective.txt")),1e-9);
    check("p5_gradient",vector(local.gradient),load(dir/(prefix+"gradient.txt")),1e-9);
    check("p5_Hessian",hessian(local.hessian),load(dir/(prefix+"hessian.txt")),1e-9);
    for(int i=0;i<3;++i)g.a[i]+=local.gradient[i];for(int i=0;i<9;++i)h.a[i]+=local.hessian[i];}
  cpu_k_correction(knots,p,&kg,&kh);
  check("K_correction_gradient",kg,load(dir/"p_initial_global_correction_gradient.txt"),1e-9);
  check("K_correction_Hessian",kh,load(dir/"p_initial_global_correction_hessian.txt"),1e-9);
  for(int i=0;i<3;++i)g.a[i]+=kg.a[i];for(int i=0;i<9;++i)h.a[i]+=kh.a[i];
  check("global_gradient",g,load(dir/"p_initial_global_raw_gradient.txt"),1e-9);
  check("global_Hessian",h,load(dir/"p_initial_global_raw_hessian.txt"),1e-9);
  auto update=cpu_newton(p,g,h);Matrix theta(3,1);theta.a={update.parameters.delta,update.parameters.alpha,update.parameters.range};
  check("Newton_update",theta,load(dir/"p_initial_joint_updated_delta_alpha_beta.txt"));
  check("Newton_step",scalar(update.accepted_step),load(dir/"p_initial_global_accepted_step.txt"));
}
}
int parity_main(int argc,char** argv){try{fs::path root=argc>1?argv[1]:"tests/fixtures";small(root/"sync_gpu");figure1(root/"figure1_reference");
  for(const auto& [name,e]:errors)std::cout<<std::setprecision(17)<<name<<" MAX_ABS="<<e.abs<<" MAX_REL="<<e.rel<<" CHECKS="<<e.checks<<'\n';
  std::cout<<"RELATIVE_ERROR_DENOMINATOR=max(1,abs(reference))\nCPU_BACKEND_PARITY=PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

int main(int argc,char** argv) {
#ifdef FEDSPATIAL_SCALAPACK
  if(argc==3&&std::string(argv[2])=="--distributed") {
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    if(provided<MPI_THREAD_FUNNELED)MPI_Abort(MPI_COMM_WORLD,2);
    int result=0;MPI_Comm comm=MPI_COMM_WORLD;
    try{cpu_worker_execute(&comm,[&]{result=parity_main(2,argv);if(result)MPI_Abort(comm,result);});}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';MPI_Abort(comm,2);}
    MPI_Finalize();return result;
  }
#endif
  return parity_main(argc,argv);
}
