#include "fedspatial/sync_cpu.hpp"
#include "fedspatial/cpu_backend.hpp"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#ifdef FEDSPATIAL_MKL_LOCAL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#elif defined(FEDSPATIAL_CRAY_LIBSCI)
#include <cblas.h>
#include <lapacke.h>
#endif
namespace fedspatial {
namespace {
void shape(const Matrix& x) {
  if(x.distributed_handle>=0)throw std::runtime_error("opaque distributed matrix used by local kernel");
  if(x.rows<0||x.cols<0||x.a.size()!=size_t(x.rows)*size_t(x.cols))throw std::runtime_error("invalid matrix storage");
}
void same_shape(const Matrix& x,const Matrix& y) {
  shape(x);shape(y);if(x.rows!=y.rows||x.cols!=y.cols)throw std::runtime_error("matrix shape mismatch");
}
}
Matrix matern15(const Matrix& x,const Matrix& y,double alpha,double range) {
  shape(x);shape(y);
  if(x.cols!=y.cols||!(alpha>0&&range>0)||!std::isfinite(alpha)||!std::isfinite(range))throw std::runtime_error("invalid Matern arguments");
  Matrix r(x.rows,y.rows);
  for(int i=0;i<x.rows;i++)for(int j=0;j<y.rows;j++) {
    double d2=0;for(int k=0;k<x.cols;k++){double q=x(i,k)-y(j,k);d2+=q*q;}
    double u=std::sqrt(3.0*d2)/range;r(i,j)=alpha*(1+u)*std::exp(-u);
  }
  return r;
}
Matrix transpose(const Matrix& x) {
  shape(x);Matrix y(x.cols,x.rows);
  for(int i=0;i<x.rows;i++)for(int j=0;j<x.cols;j++)y(j,i)=x(i,j);
  return y;
}
Matrix matmul(const Matrix& x,const Matrix& y) {
#ifdef FEDSPATIAL_SCALAPACK
  if(x.distributed_handle>=0) {
    Matrix distributed;
    if(y.distributed_handle>=0) {
      if(cpu_distributed_matmul_handles(x.distributed_handle,y.distributed_handle,
                                        x.rows,x.cols,y.cols,&distributed))return distributed;
      throw std::runtime_error("distributed matrices used outside worker service");
    }
    if(cpu_distributed_matmul_handle(x.distributed_handle,y,&distributed))return distributed;
    throw std::runtime_error("distributed matrix used outside worker service");
  }
#endif
  shape(x);shape(y);if(x.cols!=y.rows)throw std::runtime_error("matmul shape");
#ifdef FEDSPATIAL_SCALAPACK
  Matrix distributed;
  if(cpu_distributed_matmul(x,y,&distributed))return distributed;
#endif
  Matrix z(x.rows,y.cols);const int begin=cpu_row_begin(x.rows),end=cpu_row_end(x.rows);
#ifdef FEDSPATIAL_BLAS_LOCAL
  // Matrix storage is row-major.  oneMKL supplies the threaded kernel; do not
  // wrap it in an OpenMP loop, which would oversubscribe the requested cores.
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
              x.rows,y.cols,x.cols,1.0,x.a.data(),x.cols,
              y.a.data(),y.cols,0.0,z.a.data(),y.cols);
#else
#pragma omp parallel for schedule(static)
  for(int i=begin;i<end;i++)for(int k=0;k<x.cols;k++)for(int j=0;j<y.cols;j++)z(i,j)+=x(i,k)*y(k,j);
#endif
  cpu_gather_rows(z);return z;
}
Matrix matmul3(const Matrix& x,const Matrix& y,const Matrix& z) {
  shape(x);shape(y);shape(z);
  if(x.cols!=y.rows||y.cols!=z.rows)throw std::runtime_error("matmul3 shape");
#ifdef FEDSPATIAL_SCALAPACK
  Matrix distributed;
  if(cpu_distributed_matmul3(x,y,z,&distributed))return distributed;
#endif
  return matmul(matmul(x,y),z);
}
Matrix subtract(const Matrix& x,const Matrix& y) {
  same_shape(x,y);Matrix z(x.rows,x.cols);for(size_t i=0;i<x.a.size();i++)z.a[i]=x.a[i]-y.a[i];return z;
}
Matrix identity(int n){Matrix x(n,n);for(int i=0;i<n;i++)x(i,i)=1;return x;}
void add_diagonal(Matrix& x,double d){shape(x);if(x.rows!=x.cols)throw std::runtime_error("diagonal requires square matrix");for(int i=0;i<x.rows;i++)x(i,i)+=d;}
Matrix cholesky_lower(const Matrix& a) {
  shape(a);if(a.rows!=a.cols||a.rows<=0)throw std::runtime_error("Cholesky requires a nonempty square matrix");
#ifdef FEDSPATIAL_SCALAPACK
  Matrix distributed;
  if(cpu_distributed_cholesky(a,&distributed))return distributed;
#endif
#ifdef FEDSPATIAL_BLAS_LOCAL
  Matrix l=a;
  const lapack_int info=LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',l.rows,l.a.data(),l.cols);
  if(info<0)throw std::runtime_error("invalid Cholesky argument");
  if(info>0)throw std::runtime_error("non SPD");
  for(int i=0;i<l.rows;++i)for(int j=i+1;j<l.cols;++j)l(i,j)=0.0;
  return l;
#else
  int n=a.rows;Matrix l(n,n);
  for(int i=0;i<n;i++)for(int j=0;j<=i;j++) {
    double s=a(i,j);for(int q=0;q<j;q++)s-=l(i,q)*l(j,q);
    if(!std::isfinite(s))throw std::runtime_error("non-finite Cholesky input or factor");
    if(i==j){if(s<=0)throw std::runtime_error("non SPD");l(i,j)=std::sqrt(s);}else l(i,j)=s/l(j,j);
  }
  return l;
#endif
}
Matrix solve_cholesky(const Matrix& l,const Matrix& rhs) {
  shape(l);shape(rhs);if(l.rows<=0||l.rows!=l.cols||rhs.rows!=l.rows)throw std::runtime_error("SPD solve shape");
  for(int i=0;i<l.rows;++i)if(!(l(i,i)>0)||!std::isfinite(l(i,i)))throw std::runtime_error("invalid Cholesky factor");
  for(double v:rhs.a)if(!std::isfinite(v))throw std::runtime_error("non-finite SPD right hand side");
#ifdef FEDSPATIAL_SCALAPACK
  Matrix distributed;
  if(cpu_distributed_solve(l,rhs,&distributed))return distributed;
#endif
#ifdef FEDSPATIAL_BLAS_LOCAL
  Matrix out=rhs;
  const lapack_int info=LAPACKE_dpotrs(LAPACK_ROW_MAJOR,'L',l.rows,rhs.cols,
                                       l.a.data(),l.cols,out.a.data(),out.cols);
  if(info!=0)throw std::runtime_error("LAPACKE SPD solve failed: "+std::to_string(info));
  return out;
#else
  int n=l.rows;Matrix out(n,rhs.cols);
#pragma omp parallel for schedule(static)
  for(int c=0;c<rhs.cols;c++) {
    std::vector<double> y(n);
    for(int i=0;i<n;i++){double s=rhs(i,c);for(int q=0;q<i;q++)s-=l(i,q)*y[q];y[i]=s/l(i,i);}
    for(int i=n-1;i>=0;i--){double s=y[i];for(int q=i+1;q<n;q++)s-=l(q,i)*out(q,c);out(i,c)=s/l(i,i);}
  }
  return out;
#endif
}
Matrix solve_spd(const Matrix& a,const Matrix& rhs){return solve_cholesky(cholesky_lower(a),rhs);}
Matrix inverse_spd(const Matrix& a){return solve_spd(a,identity(a.rows));}
double max_abs_diff(const Matrix& x,const Matrix& y) {
  same_shape(x,y);double m=0;for(size_t i=0;i<x.a.size();i++){double d=std::abs(x.a[i]-y.a[i]);if(!std::isfinite(d))return std::numeric_limits<double>::infinity();m=std::max(m,d);}return m;
}
LocalP make_local_p(const Matrix& sites,const Matrix& knots,const Matrix& z,const Matrix& x,const Matrix& gamma,double delta,double alpha,double range) {
  LocalP q;q.K=matern15(knots,knots,alpha,range);Matrix c=matern15(sites,knots,alpha,range);
  q.B=transpose(solve_spd(q.K,transpose(c)));Matrix ss=matern15(sites,sites,alpha,range);
  q.M=subtract(ss,matmul(matmul(q.B,q.K),transpose(q.B)));for(double& v:q.M.a)v*=delta;add_diagonal(q.M,1);
  Matrix y=solve_spd(q.M,q.B);q.btMinvb=matmul(transpose(q.B),y);
  Matrix e=subtract(matmul(x,gamma),z);q.btMinve=matmul(transpose(q.B),solve_spd(q.M,e));q.Minv=inverse_spd(q.M);return q;
}
bool write_matrix(const std::string& p,const Matrix& x) {
  shape(x);std::ofstream f(p);if(!f)return false;f<<x.rows<<' '<<x.cols<<'\n'<<std::setprecision(17);
  for(double v:x.a)f<<v<<'\n';return bool(f);
}
bool read_matrix(const std::string& p,Matrix* x) {
  if(!x)return false;std::ifstream f(p);int rows=0,cols=0;
  if(!(f>>rows>>cols)||rows<=0||cols<=0)return false;
  // Validate declared storage against file size before allocating untrusted input.
  const auto start=f.tellg();f.seekg(0,std::ios::end);const auto bytes=f.tellg();f.seekg(start);
  if(start<0||bytes<start||size_t(rows)*size_t(cols)>size_t(bytes-start)/2+1)return false;
  Matrix value(rows,cols);for(double& v:value.a)if(!(f>>v)||!std::isfinite(v))return false;
  std::string extra;if(f>>extra)return false;*x=std::move(value);return true;
}
}
