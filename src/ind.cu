#include "fedspatial/gpu_context.hpp"
#include <cmath>
#include <vector>
namespace fedspatial {
namespace { void add(Matrix&a,const Matrix&b,double s=1){for(int i=0;i<a.rows*a.cols;i++)a.a[i]+=s*b.a[i];} Matrix scale(Matrix a,double s){for(double&v:a.a)v*=s;return a;} double tr(const Matrix&a){double x=0;for(int i=0;i<a.rows;i++)x+=a(i,i);return x;} double dot(const Matrix&a,const Matrix&b){double x=0;for(int i=0;i<a.rows*a.cols;i++)x+=a.a[i]*b.a[i];return x;} }
Matrix gpu_solve(GpuContext& c,const Matrix& a,const Matrix& rhs){
  const int n=a.rows,nrhs=rhs.cols; std::vector<double> ac(n*n),bc(n*nrhs),out(n*nrhs); for(int i=0;i<n;i++)for(int j=0;j<n;j++)ac[i+n*j]=a(i,j); for(int i=0;i<n;i++)for(int j=0;j<nrhs;j++)bc[i+n*j]=rhs(i,j);
  double *da,*db; cudaMalloc(&da,ac.size()*sizeof(double)); cudaMalloc(&db,bc.size()*sizeof(double)); cudaMemcpyAsync(da,ac.data(),ac.size()*sizeof(double),cudaMemcpyHostToDevice,c.stream()); cudaMemcpyAsync(db,bc.data(),bc.size()*sizeof(double),cudaMemcpyHostToDevice,c.stream());
  int work=0; cusolverDnDpotrf_bufferSize(c.cusolver(),CUBLAS_FILL_MODE_LOWER,n,da,n,&work); double* dwork; int* dinfo; cudaMalloc(&dwork,work*sizeof(double)); cudaMalloc(&dinfo,sizeof(int)); cusolverDnDpotrf(c.cusolver(),CUBLAS_FILL_MODE_LOWER,n,da,n,dwork,work,dinfo); cudaStreamSynchronize(c.stream()); double one=1.; cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,nrhs,&one,da,n,db,n); cublasDtrsm(c.cublas(),CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,n,nrhs,&one,da,n,db,n); cudaMemcpyAsync(out.data(),db,out.size()*sizeof(double),cudaMemcpyDeviceToHost,c.stream()); cudaStreamSynchronize(c.stream()); Matrix result(n,nrhs);for(int i=0;i<n;i++)for(int j=0;j<nrhs;j++)result(i,j)=out[i+n*j]; cudaFree(da);cudaFree(db);cudaFree(dwork);cudaFree(dinfo);return result;
}
GammaLocalSummary compute_ind_gamma_local(GpuContext&c,const WorkerInput&w,const GlobalParameters&p){
  const int n=w.sites.rows; Matrix R=matern_with_derivatives(c,w.sites,w.sites,p.nu2,p.alpha,p.range).value; for(int i=0;i<n;i++)R(i,i)+=1./p.delta; Matrix rhs(n,w.x.cols+1);for(int i=0;i<n;i++){rhs(i,0)=w.z(i,0);for(int j=0;j<w.x.cols;j++)rhs(i,j+1)=w.x(i,j);} Matrix solved=gpu_solve(c,R,rhs), Wz(n,1),WX(n,w.x.cols);for(int i=0;i<n;i++){Wz(i,0)=solved(i,0);for(int j=0;j<w.x.cols;j++)WX(i,j)=solved(i,j+1);} return {matmul(transpose(w.x),Wz),matmul(transpose(w.x),WX)};
}
IndDeltaThetaLocalResult compute_ind_delta_theta_local(GpuContext&c,const WorkerInput&w,const GlobalParameters&p){
  const int n=w.sites.rows; Matrix C=matern_with_derivatives(c,w.sites,w.sites,p.nu2,p.alpha,p.range).value,R=C; for(int i=0;i<n;i++)R(i,i)+=1./p.delta; Matrix W=gpu_solve(c,R,identity(n)), e=subtract(matmul(w.x,p.gamma),w.z), u=matmul(W,e);
  IndDeltaThetaLocalResult out; double ld=0; Matrix L=cholesky_lower(R); for(int i=0;i<n;i++)ld+=2*std::log(L(i,i)); out.objective=(ld+dot(e,u))/n;
  const auto deriv=matern_with_derivatives(c,w.sites,w.sites,p.nu2,p.alpha,p.range); Matrix I=identity(n), Ri[3]={scale(I,-1./(p.delta*p.delta)),deriv.alpha,deriv.range};
  Matrix Rij[3][3]={{scale(I,2./(p.delta*p.delta*p.delta)),Matrix(n,n),Matrix(n,n)},{Matrix(n,n),deriv.aa,deriv.ar},{Matrix(n,n),deriv.ra,deriv.rr}};
  for(int i=0;i<3;i++){out.gradient[i]=(tr(matmul(W,Ri[i]))-dot(u,matmul(Ri[i],u)))/n;for(int j=i;j<3;j++){double h=tr(matmul(W,Rij[i][j]))-tr(matmul(matmul(matmul(W,Ri[j]),W),Ri[i]))+2*dot(u,matmul(matmul(Ri[j],W),matmul(Ri[i],u)))-dot(u,matmul(Rij[i][j],u));out.hessian[3*i+j]=out.hessian[3*j+i]=h/n;}}
  return out;
}
PSynchronousRoundResult run_ind_synchronous_round(GpuContext& c,const std::vector<WorkerInput>& ws,const GlobalParameters& in){
  const int J=ws.size(),q=in.gamma.rows,n=ws[0].sites.rows; GlobalParameters p=in; Matrix gg(q,1),xx(q,q);
  for(const auto&w:ws){auto s=compute_ind_gamma_local(c,w,p);add(gg,s.gamma);add(xx,s.xx);} for(double&v:gg.a)v/=J; for(double&v:xx.a)v/=J; p.gamma=solve_spd(xx,gg);
  Matrix g(3,1),h(3,3); for(const auto&w:ws){auto s=compute_ind_delta_theta_local(c,w,p);for(int i=0;i<3;i++){g(i,0)+=s.gradient[i];for(int j=0;j<3;j++)h(i,j)+=s.hessian[3*i+j];}} for(double&v:g.a)v/=J;for(double&v:h.a)v/=J;
  // Symmetric Jacobi eigendecomposition with the same abs-eigenvalue floor as Python.
  Matrix v=identity(3),a=h;for(int it=0;it<64;it++){int pi=0,pj=1;double mx=std::abs(a(0,1));for(int i=0;i<3;i++)for(int j=i+1;j<3;j++)if(std::abs(a(i,j))>mx){mx=std::abs(a(i,j));pi=i;pj=j;}if(mx<1e-15)break;double ph=.5*std::atan2(2*a(pi,pj),a(pj,pj)-a(pi,pi)),co=std::cos(ph),si=std::sin(ph);for(int k=0;k<3;k++){double x=a(pi,k),y=a(pj,k);a(pi,k)=co*x-si*y;a(pj,k)=si*x+co*y;}for(int k=0;k<3;k++){double x=a(k,pi),y=a(k,pj);a(k,pi)=co*x-si*y;a(k,pj)=si*x+co*y;double z=v(k,pi),t=v(k,pj);v(k,pi)=co*z-si*t;v(k,pj)=si*z+co*t;}}
  Matrix mod(3,3);for(int k=0;k<3;k++){double ev=std::max(std::abs(a(k,k)),.01);for(int i=0;i<3;i++)for(int j=0;j<3;j++)mod(i,j)+=v(i,k)*ev*v(j,k);}Matrix d=solve_spd(mod,g);double step=1.;double nd=p.delta-step*d(0,0),na=p.alpha-step*d(1,0),nr=p.range-step*d(2,0);int halves=0;while(nd<=0||na<=0||nr<=0){step/=2;halves++;nd=p.delta-step*d(0,0);na=p.alpha-step*d(1,0);nr=p.range-step*d(2,0);}p.delta=nd;p.alpha=na;p.range=nr;return {p,step,halves};
}
}
