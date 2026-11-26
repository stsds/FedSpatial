#include "fedspatial/cpu_backend.hpp"
#include <mpi.h>
#ifdef FEDSPATIAL_CRAY_LIBSCI
#include <cblas.h>
#include <lapacke.h>
#else
#include <mkl_scalapack.h>
#include <mkl_cblas.h>
#include <mkl_blacs.h>
#endif
#include <algorithm>
#include <numeric>
#include <limits>
#include <memory>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <array>
#include <cmath>

// BLACS's C communicator entry point is not declared in mkl_blacs.h.
extern "C" {
int Csys2blacs_handle(MPI_Comm);
void Cblacs_gridinit(int*,const char*,int,int);
void Cblacs_gridexit(int);
#ifdef FEDSPATIAL_CRAY_LIBSCI
void descinit_(int*,const int*,const int*,const int*,const int*,const int*,
               const int*,const int*,const int*,int*);
void pdpotrf_(const char*,const int*,double*,const int*,const int*,const int*,int*);
void pdpotrs_(const char*,const int*,const int*,const double*,const int*,const int*,
              const int*,double*,const int*,const int*,const int*,int*);
#endif
}
namespace fedspatial {
namespace {
MPI_Comm service_comm=MPI_COMM_NULL;
int service_rank=0,service_size=1,context=-1;
bool active=false;
bool trace(){return std::getenv("FEDSPATIAL_TRACE_DISTRIBUTED")!=nullptr;}
std::unordered_map<int,Matrix> pinned_rows;
std::unordered_map<const double*,int> leader_pins;
int next_pin_id=1;
Matrix site_context;
int site_nu2=3;
double site_alpha=1.0,site_range=1.0;
std::array<std::uint64_t,8> operation_bytes{};
std::array<std::uint64_t,8> operation_calls{};
std::uint64_t operation_begin(){return cpu_communication_bytes();}
void operation_end(int op,std::uint64_t before) {
  ++operation_calls[op];operation_bytes[op]+=cpu_communication_bytes()-before;
}
void check(int code) {
  if(code!=MPI_SUCCESS)throw std::runtime_error("distributed CPU MPI operation failed");
}
int count(size_t n) {
  if(n>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("distributed CPU MPI count overflow");
  return int(n);
}
void command(int op,int m,int k,int n,int q=0) {
  int h[5]={op,m,k,n,q};check(MPI_Bcast(h,5,MPI_INT,0,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(sizeof(h)*(service_size-1));
}
Matrix distribute_rows(const Matrix* x,int m,int n) {
  std::vector<int> counts(service_size),displacements(service_size);
  for(int rank=0;rank<service_size;++rank) {
    const int begin=int((long long)m*rank/service_size);
    const int end=int((long long)m*(rank+1)/service_size);
    counts[rank]=count(size_t(end-begin)*n);
    displacements[rank]=count(size_t(begin)*n);
  }
  const int rows=int((long long)m*(service_rank+1)/service_size)-
                 int((long long)m*service_rank/service_size);
  Matrix local(rows,n);
  check(MPI_Scatterv(x?x->a.data():nullptr,counts.data(),displacements.data(),MPI_DOUBLE,
                     local.a.data(),counts[service_rank],MPI_DOUBLE,0,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(
      std::uint64_t(std::accumulate(counts.begin(),counts.end(),0LL)-counts[0])*sizeof(double));
  return local;
}
void pin_rows(const Matrix* x,int m,int n,int id) {
  pinned_rows.insert_or_assign(id,distribute_rows(x,m,n));
}
void unpin_rows(int id){pinned_rows.erase(id);}
Matrix multiply3(const Matrix* x,const Matrix* y,const Matrix* z,
                 int m,int k,int p,int n,int pin_id=-1) {
  std::vector<int> xc(service_size),xd(service_size),zc(service_size),zd(service_size);
  for(int r=0;r<service_size;++r) {
    int b=int((long long)m*r/service_size),e=int((long long)m*(r+1)/service_size);
    xc[r]=count(size_t(e-b)*k);xd[r]=count(size_t(b)*k);
    zc[r]=count(size_t(e-b)*n);zd[r]=count(size_t(b)*n);
  }
  const int rows=int((long long)m*(service_rank+1)/service_size)-int((long long)m*service_rank/service_size);
  Matrix transient;
  const Matrix* local_x=nullptr;
  if(pin_id>=0) {
    auto it=pinned_rows.find(pin_id);
    if(it==pinned_rows.end()||it->second.rows!=rows||it->second.cols!=k)
      throw std::runtime_error("invalid distributed pinned operand");
    local_x=&it->second;
  } else {transient=distribute_rows(x,m,k);local_x=&transient;}
  Matrix local_xy(rows,p),local_z(rows,n);
  const int panel_rows=64;
  std::vector<double> panel(size_t(panel_rows)*std::max(p,n));
  for(int b=0;b<k;b+=panel_rows) {
    int end=std::min(k,b+panel_rows),elements=count(size_t(end-b)*p);
    if(service_rank==0)std::copy_n(y->a.data()+size_t(b)*p,elements,panel.data());
    check(MPI_Bcast(panel.data(),elements,MPI_DOUBLE,0,service_comm));
    if(service_rank==0)cpu_add_communication_bytes(
        std::uint64_t(elements)*sizeof(double)*(service_size-1));
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                rows,p,end-b,1.0,local_x->a.data()+b,k,
                panel.data(),p,1.0,local_xy.a.data(),p);
  }
  for(int b=0;b<p;b+=panel_rows) {
    int end=std::min(p,b+panel_rows),elements=count(size_t(end-b)*n);
    if(service_rank==0)std::copy_n(z->a.data()+size_t(b)*n,elements,panel.data());
    check(MPI_Bcast(panel.data(),elements,MPI_DOUBLE,0,service_comm));
    if(service_rank==0)cpu_add_communication_bytes(
        std::uint64_t(elements)*sizeof(double)*(service_size-1));
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                rows,n,end-b,1.0,local_xy.a.data()+b,p,
                panel.data(),n,1.0,local_z.a.data(),n);
  }
  Matrix result=service_rank==0?Matrix(m,n):Matrix();
  check(MPI_Gatherv(local_z.a.data(),zc[service_rank],MPI_DOUBLE,result.a.data(),
                    zc.data(),zd.data(),MPI_DOUBLE,0,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(
      std::uint64_t(std::accumulate(zc.begin(),zc.end(),0LL)-zc[0])*sizeof(double));
  return result;
}
Matrix subtract_triples(const Matrix* initial,const Matrix* const* left,
                        const Matrix* const* middle,const Matrix* const* right,
                        int n,int m,int terms,int retain_id=-1,int site_kind=-1) {
  std::vector<int> lc(service_size),ld(service_size),oc(service_size),od(service_size);
  for(int rank=0;rank<service_size;++rank) {
    const int begin=int((long long)n*rank/service_size);
    const int end=int((long long)n*(rank+1)/service_size);
    lc[rank]=count(size_t(end-begin)*m);ld[rank]=count(size_t(begin)*m);
    oc[rank]=count(size_t(end-begin)*n);od[rank]=count(size_t(begin)*n);
  }
  const int rows=int((long long)n*(service_rank+1)/service_size)-
                 int((long long)n*service_rank/service_size);
  Matrix local_out(rows,n),local_left(rows,m),local_product(rows,n),local_lk(rows,m);
  if(site_kind<0) {
    check(MPI_Scatterv(initial?initial->a.data():nullptr,oc.data(),od.data(),MPI_DOUBLE,
                       local_out.a.data(),oc[service_rank],MPI_DOUBLE,0,service_comm));
    if(service_rank==0)cpu_add_communication_bytes(
        std::uint64_t(std::accumulate(oc.begin(),oc.end(),0LL)-oc[0])*sizeof(double));
  } else if(site_kind!=0) {
    if(site_context.rows!=n||site_context.cols!=2)
      throw std::runtime_error("distributed site context shape");
    const int row_begin=int((long long)n*service_rank/service_size);
#pragma omp parallel for schedule(static)
    for(int local_i=0;local_i<rows;++local_i)for(int j=0;j<n;++j) {
      const int i=row_begin+local_i;
      double dx=site_context(i,0)-site_context(j,0);
      double dy=site_context(i,1)-site_context(j,1);
      double u=std::sqrt(dx*dx+dy*dy)/site_range,e,p,rb,rbb;
      if(site_nu2==1){e=std::exp(-u);p=e;rb=e*u/site_range;rbb=site_alpha*e*(u*u-2*u)/(site_range*site_range);}
      else if(site_nu2==3){u*=std::sqrt(3.);e=std::exp(-u);p=(1+u)*e;rb=e*u*u/site_range;rbb=site_alpha*e*(u*u*u-3*u*u)/(site_range*site_range);}
      else{u*=std::sqrt(5.);e=std::exp(-u);p=(1+u+u*u/3)*e;rb=e*(u*u+u*u*u)/(3*site_range);rbb=site_alpha*e*(u*u*u*u-3*u*u*u-3*u*u)/(3*site_range*site_range);}
      local_out(local_i,j)=site_kind==1?p:(site_kind==2?site_alpha*rb:(site_kind==3?rb:rbb));
    }
  }
  const int panel_rows=64;
  std::vector<double> panel(size_t(panel_rows)*std::max(m,n));
  for(int term=0;term<terms;++term) {
    std::fill(local_product.a.begin(),local_product.a.end(),0.0);
    std::fill(local_lk.a.begin(),local_lk.a.end(),0.0);
    check(MPI_Scatterv(left?left[term]->a.data():nullptr,lc.data(),ld.data(),MPI_DOUBLE,
                       local_left.a.data(),lc[service_rank],MPI_DOUBLE,0,service_comm));
    if(service_rank==0)cpu_add_communication_bytes(
        std::uint64_t(std::accumulate(lc.begin(),lc.end(),0LL)-lc[0])*sizeof(double));
    for(int b=0;b<m;b+=panel_rows) {
      const int end=std::min(m,b+panel_rows),elements=count(size_t(end-b)*m);
      if(service_rank==0)
        std::copy_n(middle[term]->a.data()+size_t(b)*m,elements,panel.data());
      check(MPI_Bcast(panel.data(),elements,MPI_DOUBLE,0,service_comm));
      if(service_rank==0)cpu_add_communication_bytes(
          std::uint64_t(elements)*sizeof(double)*(service_size-1));
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                  rows,m,end-b,1.0,local_left.a.data()+b,m,
                  panel.data(),m,1.0,local_lk.a.data(),m);
    }
    // Stream panels of transpose(right) without materializing that matrix on
    // the leader. right is n-by-m, so panel(q,i)=right(i,q).
    for(int b=0;b<m;b+=panel_rows) {
      const int end=std::min(m,b+panel_rows),panel_height=end-b;
      const int elements=count(size_t(panel_height)*n);
      if(service_rank==0) for(int q=b;q<end;++q) for(int i=0;i<n;++i)
        panel[size_t(q-b)*n+i]=(*right[term])(i,q);
      check(MPI_Bcast(panel.data(),elements,MPI_DOUBLE,0,service_comm));
      if(service_rank==0)cpu_add_communication_bytes(
          std::uint64_t(elements)*sizeof(double)*(service_size-1));
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                  rows,n,panel_height,1.0,local_lk.a.data()+b,m,
                  panel.data(),n,1.0,local_product.a.data(),n);
    }
    for(size_t i=0;i<local_out.a.size();++i)local_out.a[i]-=local_product.a[i];
  }
  if(retain_id>=0) {
    pinned_rows.insert_or_assign(retain_id,std::move(local_out));return Matrix();
  }
  Matrix result=service_rank==0?Matrix(n,n):Matrix();
  check(MPI_Gatherv(local_out.a.data(),oc[service_rank],MPI_DOUBLE,result.a.data(),
                    oc.data(),od.data(),MPI_DOUBLE,0,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(
      std::uint64_t(std::accumulate(oc.begin(),oc.end(),0LL)-oc[0])*sizeof(double));
  return result;
}
void scale_handle(int id,double scale) {
  auto it=pinned_rows.find(id);if(it==pinned_rows.end())
    throw std::runtime_error("invalid distributed scale handle");
  for(double& value:it->second.a)value*=scale;
}
double trace_product_handle(const Matrix* matrix,int matrix_id,int other_id,int n) {
  auto other=pinned_rows.find(other_id);if(other==pinned_rows.end())
    throw std::runtime_error("invalid distributed trace-product handle");
  Matrix transient;
  const Matrix* local_matrix=nullptr;
  if(matrix_id>=0) {
    auto it=pinned_rows.find(matrix_id);if(it==pinned_rows.end())
      throw std::runtime_error("invalid distributed trace-product matrix");
    local_matrix=&it->second;
  } else {transient=distribute_rows(matrix,n,n);local_matrix=&transient;}
  if(local_matrix->rows!=other->second.rows||local_matrix->cols!=n||other->second.cols!=n)
    throw std::runtime_error("distributed trace-product shape");
  // Both matrices are symmetric in this Block-3 use, so trace(A B) is their
  // Frobenius inner product. Preserve row-major accumulation locally.
  double local=0.0;for(size_t i=0;i<local_matrix->a.size();++i)
    local+=local_matrix->a[i]*other->second.a[i];
  double global=0.0;check(MPI_Reduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,0,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(
      std::uint64_t(service_size-1)*sizeof(double));
  return global;
}
int clone_scale_handle(int source_id,int target_id,double scale) {
  auto source=pinned_rows.find(source_id);if(source==pinned_rows.end())
    throw std::runtime_error("invalid distributed clone source");
  Matrix copy=source->second;for(double& value:copy.a)value*=scale;
  pinned_rows.insert_or_assign(target_id,std::move(copy));return target_id;
}
int zero_handle(int id,int n) {
  const int rows=int((long long)n*(service_rank+1)/service_size)-
                 int((long long)n*service_rank/service_size);
  pinned_rows.insert_or_assign(id,Matrix(rows,n));return id;
}
Matrix multiply_handles(int left_id,int right_id,int m,int k,int n) {
  auto left=pinned_rows.find(left_id),right=pinned_rows.find(right_id);
  if(left==pinned_rows.end()||right==pinned_rows.end())
    throw std::runtime_error("invalid distributed handle product");
  const int rows=left->second.rows;
  Matrix local_result(rows,n);
  std::vector<double> panel;
  for(int owner=0;owner<service_size;++owner) {
    const int begin=int((long long)k*owner/service_size);
    const int end=int((long long)k*(owner+1)/service_size);
    const int elements=count(size_t(end-begin)*n);
    panel.resize(elements);
    if(service_rank==owner) {
      if(right->second.rows!=end-begin||right->second.cols!=n)
        throw std::runtime_error("distributed right-handle shape");
      std::copy_n(right->second.a.data(),elements,panel.data());
    }
    check(MPI_Bcast(panel.data(),elements,MPI_DOUBLE,owner,service_comm));
    if(service_rank==0)cpu_add_communication_bytes(
        std::uint64_t(elements)*sizeof(double)*(service_size-1));
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                rows,n,end-begin,1.0,left->second.a.data()+begin,k,
                panel.data(),n,1.0,local_result.a.data(),n);
  }
  std::vector<int> counts(service_size),displacements(service_size);
  for(int rank=0;rank<service_size;++rank) {
    const int begin=int((long long)m*rank/service_size);
    const int end=int((long long)m*(rank+1)/service_size);
    counts[rank]=count(size_t(end-begin)*n);displacements[rank]=count(size_t(begin)*n);
  }
  Matrix result=service_rank==0?Matrix(m,n):Matrix();
  check(MPI_Gatherv(local_result.a.data(),counts[service_rank],MPI_DOUBLE,
                    result.a.data(),counts.data(),displacements.data(),MPI_DOUBLE,0,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(
      std::uint64_t(std::accumulate(counts.begin(),counts.end(),0LL)-counts[0])*sizeof(double));
  return result;
}
void set_site_context(const Matrix* sites,int n,int nu2,double alpha=0,double range=0) {
  double parameters[2]={alpha,range};
  check(MPI_Bcast(parameters,2,MPI_DOUBLE,0,service_comm));
  site_nu2=nu2;site_alpha=parameters[0];site_range=parameters[1];
  site_context=Matrix(n,2);
  if(service_rank==0)site_context=*sites;
  check(MPI_Bcast(site_context.a.data(),count(site_context.a.size()),MPI_DOUBLE,0,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(
      std::uint64_t(service_size-1)*(2*sizeof(double)+site_context.a.size()*sizeof(double)));
}
Matrix multiply(const Matrix* x,const Matrix* y,int m,int k,int n,int pin_id=-1) {
  std::vector<int> xc(service_size),xd(service_size),zc(service_size),zd(service_size);
  for(int r=0;r<service_size;++r) {
    int b=int((long long)m*r/service_size),e=int((long long)m*(r+1)/service_size);
    xc[r]=count(size_t(e-b)*k);xd[r]=count(size_t(b)*k);
    zc[r]=count(size_t(e-b)*n);zd[r]=count(size_t(b)*n);
  }
  const int rows=int((long long)m*(service_rank+1)/service_size)-int((long long)m*service_rank/service_size);
  Matrix transient;
  const Matrix* local_x=nullptr;
  if(pin_id>=0) {
    auto it=pinned_rows.find(pin_id);
    if(it==pinned_rows.end()||it->second.rows!=rows||it->second.cols!=k)
      throw std::runtime_error("invalid distributed pinned operand");
    local_x=&it->second;
  } else {transient=distribute_rows(x,m,k);local_x=&transient;}
  Matrix local_z(rows,n);
  // Stream RHS row panels. No follower allocates a full RHS or result.
  const int panel_rows=64;
  std::vector<double> panel(size_t(std::min(panel_rows,k))*n);
  for(int b=0;b<k;b+=panel_rows) {
    int end=std::min(k,b+panel_rows),elements=count(size_t(end-b)*n);
    if(service_rank==0)std::copy_n(y->a.data()+size_t(b)*n,elements,panel.data());
    check(MPI_Bcast(panel.data(),elements,MPI_DOUBLE,0,service_comm));
    if(service_rank==0)cpu_add_communication_bytes(
        std::uint64_t(elements)*sizeof(double)*(service_size-1));
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                rows,n,end-b,1.0,local_x->a.data()+b,k,
                panel.data(),n,1.0,local_z.a.data(),n);
  }
  Matrix result=service_rank==0?Matrix(m,n):Matrix();
  check(MPI_Gatherv(local_z.a.data(),zc[service_rank],MPI_DOUBLE,result.a.data(),
                    zc.data(),zd.data(),MPI_DOUBLE,0,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(
      std::uint64_t(std::accumulate(zc.begin(),zc.end(),0LL)-zc[0])*sizeof(double));
  return result;
}
double trace4_symmetric(const Matrix* right1,const Matrix* right2,
                        int left1_id,int left2_id,int n) {
  auto first=pinned_rows.find(left1_id),second=pinned_rows.find(left2_id);
  if(first==pinned_rows.end()||second==pinned_rows.end())
    throw std::runtime_error("invalid distributed trace operands");
  const int rows=first->second.rows;
  if(second->second.rows!=rows||first->second.cols!=n||second->second.cols!=n)
    throw std::runtime_error("distributed trace shape");
  Matrix product1(rows,n),product2_transpose(rows,n);
  const int panel_rows=64;
  std::vector<double> panel(size_t(panel_rows)*n);
  auto product=[&](const Matrix* right,const Matrix& left,Matrix& out) {
    for(int b=0;b<n;b+=panel_rows) {
      const int end=std::min(n,b+panel_rows),elements=count(size_t(end-b)*n);
      if(service_rank==0)std::copy_n(right->a.data()+size_t(b)*n,elements,panel.data());
      check(MPI_Bcast(panel.data(),elements,MPI_DOUBLE,0,service_comm));
      if(service_rank==0)cpu_add_communication_bytes(
          std::uint64_t(elements)*sizeof(double)*(service_size-1));
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                  rows,n,end-b,1.0,left.a.data()+b,n,
                  panel.data(),n,1.0,out.a.data(),n);
    }
  };
  // trace((A B)(C D)) = <A B, (C D)^T>_F.  For the symmetric
  // operands used by Block 3, (C D)^T = D C.
  product(right1,first->second,product1);
  product(right2,second->second,product2_transpose);
  double local=0.0;for(size_t i=0;i<product1.a.size();++i)
    local+=product1.a[i]*product2_transpose.a[i];
  double global=0.0;check(MPI_Reduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,0,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(
      std::uint64_t(service_size-1)*sizeof(double));
  return global;
}
double trace4_handles(int a_id,int b_id,int c_id,int d_id,int n) {
  auto local_product=[&](int left_id,int right_id) {
    auto left=pinned_rows.find(left_id),right=pinned_rows.find(right_id);
    if(left==pinned_rows.end()||right==pinned_rows.end())
      throw std::runtime_error("invalid distributed trace4 handle");
    Matrix out(left->second.rows,n);std::vector<double> panel;
    for(int owner=0;owner<service_size;++owner) {
      const int begin=int((long long)n*owner/service_size);
      const int end=int((long long)n*(owner+1)/service_size);
      const int elements=count(size_t(end-begin)*n);panel.resize(elements);
      if(service_rank==owner)std::copy_n(right->second.a.data(),elements,panel.data());
      check(MPI_Bcast(panel.data(),elements,MPI_DOUBLE,owner,service_comm));
      if(service_rank==0)cpu_add_communication_bytes(
          std::uint64_t(elements)*sizeof(double)*(service_size-1));
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                  left->second.rows,n,end-begin,1.0,
                  left->second.a.data()+begin,n,panel.data(),n,
                  1.0,out.a.data(),n);
    }
    return out;
  };
  Matrix first=local_product(a_id,b_id);
  // (C D)^T = D C for the symmetric Block-3 operands.
  Matrix second_transpose=local_product(d_id,c_id);
  double local=0.0;for(size_t i=0;i<first.a.size();++i)
    local+=first.a[i]*second_transpose.a[i];
  double global=0.0;check(MPI_Reduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,0,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(
      std::uint64_t(service_size-1)*sizeof(double));
  return global;
}
struct ColumnMatrix {
  int rows,cols,block;
  std::vector<int> columns;
  std::vector<double> local;
  int desc[9];
  int owner(int j)const{return (j/block)%service_size;}
  ColumnMatrix(const Matrix* a,int m,int n,int b):rows(m),cols(n),block(b),columns(service_size) {
    for(int j=0;j<cols;++j)++columns[owner(j)];
    local.resize(std::max(size_t(1),size_t(rows)*columns[service_rank]));
    // Pack one rank at a time: leader staging also scales as 1/P.
    for(int r=0;r<service_size;++r) {
      std::vector<double> packed;
      if(service_rank==0) {
        packed.resize(size_t(rows)*columns[r]);int c=0;
        for(int j=0;j<cols;++j)if(owner(j)==r) {
          for(int i=0;i<rows;++i)packed[size_t(c)*rows+i]=(*a)(i,j);++c;
        }
        if(r==0)std::copy(packed.begin(),packed.end(),local.begin());
        else {check(MPI_Send(packed.data(),count(packed.size()),MPI_DOUBLE,r,17,service_comm));
          cpu_add_communication_bytes(std::uint64_t(packed.size())*sizeof(double));}
      } else if(service_rank==r)
        check(MPI_Recv(local.data(),count(size_t(rows)*columns[r]),MPI_DOUBLE,0,17,service_comm,MPI_STATUS_IGNORE));
    }
    int info=0,zero=0,lda=std::max(1,rows);
    descinit_(desc,&rows,&cols,&block,&block,&zero,&zero,&context,&lda,&info);
    if(info!=0)throw std::runtime_error("ScaLAPACK descriptor failed: "+std::to_string(info));
  }
  // Construct a distributed identity without materializing it on the leader.
  ColumnMatrix(int n,int b):rows(n),cols(n),block(b),columns(service_size) {
    for(int j=0;j<cols;++j)++columns[owner(j)];
    local.assign(std::max(size_t(1),size_t(rows)*columns[service_rank]),0.0);
    int c=0;
    for(int j=0;j<cols;++j)if(owner(j)==service_rank) {
      local[size_t(c)*rows+j]=1.0;++c;
    }
    int info=0,zero=0,lda=std::max(1,rows);
    descinit_(desc,&rows,&cols,&block,&block,&zero,&zero,&context,&lda,&info);
    if(info!=0)throw std::runtime_error("ScaLAPACK descriptor failed: "+std::to_string(info));
  }
  Matrix gather(bool lower=false)const {
    Matrix result=service_rank==0?Matrix(rows,cols):Matrix();
    for(int r=0;r<service_size;++r) {
      if(service_rank==r&&r!=0)check(MPI_Send(local.data(),count(size_t(rows)*columns[r]),MPI_DOUBLE,0,18,service_comm));
      if(service_rank==0) {
        std::vector<double> packed;
        const double* data=local.data();
        if(r!=0) {
          packed.resize(size_t(rows)*columns[r]);
          check(MPI_Recv(packed.data(),count(packed.size()),MPI_DOUBLE,r,18,service_comm,MPI_STATUS_IGNORE));data=packed.data();
          cpu_add_communication_bytes(std::uint64_t(packed.size())*sizeof(double));
        }
        int c=0;
        for(int j=0;j<cols;++j)if(owner(j)==r) {
          for(int i=lower?j:0;i<rows;++i)result(i,j)=data[size_t(c)*rows+i];++c;
        }
      }
    }
    return result;
  }
  Matrix row_blocks()const {
    const int local_begin=int((long long)rows*service_rank/service_size);
    const int local_end=int((long long)rows*(service_rank+1)/service_size);
    const int local_rows=local_end-local_begin;
    std::vector<int> sendcounts(service_size),senddisplacements(service_size);
    std::vector<int> recvcounts(service_size),recvdisplacements(service_size);
    int send_total=0,recv_total=0;
    for(int rank=0;rank<service_size;++rank) {
      const int begin=int((long long)rows*rank/service_size);
      const int end=int((long long)rows*(rank+1)/service_size);
      sendcounts[rank]=count(size_t(end-begin)*columns[service_rank]);
      senddisplacements[rank]=send_total;send_total+=sendcounts[rank];
      recvcounts[rank]=count(size_t(local_rows)*columns[rank]);
      recvdisplacements[rank]=recv_total;recv_total+=recvcounts[rank];
    }
    std::vector<double> send(send_total),recv(recv_total);
    for(int rank=0;rank<service_size;++rank) {
      const int begin=int((long long)rows*rank/service_size);
      const int end=int((long long)rows*(rank+1)/service_size);
      double* destination=send.data()+senddisplacements[rank];
      for(int c=0;c<columns[service_rank];++c)
        for(int i=begin;i<end;++i)*destination++=local[size_t(c)*rows+i];
    }
    check(MPI_Alltoallv(send.data(),sendcounts.data(),senddisplacements.data(),MPI_DOUBLE,
                        recv.data(),recvcounts.data(),recvdisplacements.data(),MPI_DOUBLE,
                        service_comm));
    if(service_rank==0)cpu_add_communication_bytes(
      std::uint64_t(rows)*cols*sizeof(double)*(service_size-1)/service_size);
    Matrix result(local_rows,cols);
    for(int source=0;source<service_size;++source) {
      const double* input=recv.data()+recvdisplacements[source];int c=0;
      for(int j=0;j<cols;++j)if(owner(j)==source) {
        for(int i=0;i<local_rows;++i)result(i,j)=input[size_t(c)*local_rows+i];++c;
      }
    }
    return result;
  }
};
// A Cholesky factor is normally followed by several solves with the exact
// returned Matrix object. Keep its distributed columns resident for the
// duration of cpu_worker_execute instead of redistributing the same n-by-n
// factor before every right-hand side.
std::unique_ptr<ColumnMatrix> factor_cache;
const double* leader_factor_pointer=nullptr;
int block_size(int n){return std::max(1,std::min(64,n/(2*service_size)));}
void factor_status(int info,const char* operation) {
  int min_info=0,max_info=0;
  check(MPI_Allreduce(&info,&min_info,1,MPI_INT,MPI_MIN,service_comm));
  check(MPI_Allreduce(&info,&max_info,1,MPI_INT,MPI_MAX,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(
      4ULL*std::uint64_t(service_size-1)*sizeof(int));
  if(min_info!=0||max_info!=0)throw std::runtime_error(std::string("ScaLAPACK ")+operation+" failed: "+std::to_string(min_info)+","+std::to_string(max_info));
}
Matrix factor(const Matrix* a,int n) {
  auto data=std::make_unique<ColumnMatrix>(a,n,n,block_size(n));
  int one=1,info=0;
  pdpotrf_("L",&n,data->local.data(),&one,&one,data->desc,&info);
  factor_status(info,"pdpotrf");
  Matrix gathered=data->gather(true);
  factor_cache=std::move(data);
  return gathered;
}
Matrix solve(const Matrix* l,const Matrix* rhs,int n,int nrhs,bool reuse) {
  const int block=block_size(n);
  std::unique_ptr<ColumnMatrix> temporary;
  if(!reuse||!factor_cache||factor_cache->rows!=n||factor_cache->cols!=n)
    temporary=std::make_unique<ColumnMatrix>(l,n,n,block);
  ColumnMatrix& factor_data=temporary?*temporary:*factor_cache;
  ColumnMatrix b(rhs,n,nrhs,block);
  int one=1,info=0;
  pdpotrs_("L",&n,&nrhs,factor_data.local.data(),&one,&one,factor_data.desc,
           b.local.data(),&one,&one,b.desc,&info);
  factor_status(info,"pdpotrs");return b.gather();
}
int solve_identity_handle(const Matrix* l,int n,bool reuse,int id) {
  const int block=block_size(n);
  std::unique_ptr<ColumnMatrix> temporary;
  if(!reuse||!factor_cache||factor_cache->rows!=n||factor_cache->cols!=n)
    temporary=std::make_unique<ColumnMatrix>(l,n,n,block);
  ColumnMatrix& factor_data=temporary?*temporary:*factor_cache;
  ColumnMatrix b(n,block);
  int one=1,info=0;
  pdpotrs_("L",&n,&n,factor_data.local.data(),&one,&one,factor_data.desc,
           b.local.data(),&one,&one,b.desc,&info);
  factor_status(info,"pdpotrs");
  pinned_rows.insert_or_assign(id,b.row_blocks());return id;
}
}
void cpu_worker_execute(const void* communicator,const std::function<void()>& f) {
  service_comm=*static_cast<const MPI_Comm*>(communicator);
  check(MPI_Comm_rank(service_comm,&service_rank));check(MPI_Comm_size(service_comm,&service_size));
  if(service_size==1){f();service_comm=MPI_COMM_NULL;return;}
  context=Csys2blacs_handle(service_comm);
  Cblacs_gridinit(&context,"R",1,service_size);
  active=true;
  if(service_rank==0){f();command(0,0,0,0);}
  else for(;;) {
    int h[5];check(MPI_Bcast(h,5,MPI_INT,0,service_comm));
    if(trace()){std::fprintf(stderr,"service rank %d op %d %d %d %d %d begin\n",service_rank,h[0],h[1],h[2],h[3],h[4]);std::fflush(stderr);}
    if(h[0]==0)break;
    if(h[0]==1)multiply(nullptr,nullptr,h[1],h[2],h[3],h[4]-1);
    else if(h[0]==2)factor(nullptr,h[1]);
    else if(h[0]==3)solve(nullptr,nullptr,h[1],h[3],h[2]!=0);
    else if(h[0]==4) {
      int pin_header=0;check(MPI_Bcast(&pin_header,1,MPI_INT,0,service_comm));
      multiply3(nullptr,nullptr,nullptr,h[1],h[2],h[3],h[4],pin_header-1);
    }
    else if(h[0]==5)subtract_triples(nullptr,nullptr,nullptr,nullptr,h[1],h[2],h[3]);
    else if(h[0]==6)pin_rows(nullptr,h[1],h[2],h[3]);
    else if(h[0]==7)unpin_rows(h[1]);
    else if(h[0]==8) {
      int n=0;check(MPI_Bcast(&n,1,MPI_INT,0,service_comm));
      trace4_handles(h[1],h[2],h[3],h[4],n);
    }
    else if(h[0]==9)subtract_triples(nullptr,nullptr,nullptr,nullptr,h[1],h[2],h[3],h[4]);
    else if(h[0]==10)multiply(nullptr,nullptr,h[2],h[2],h[3],h[1]);
    else if(h[0]==11)trace_product_handle(nullptr,h[1],h[2],h[3]);
    else if(h[0]==12) {
      double scale=0;check(MPI_Bcast(&scale,1,MPI_DOUBLE,0,service_comm));scale_handle(h[1],scale);
    }
    else if(h[0]==13) {
      double scale=0;check(MPI_Bcast(&scale,1,MPI_DOUBLE,0,service_comm));
      clone_scale_handle(h[1],h[2],scale);
    }
    else if(h[0]==14)multiply_handles(h[1],h[2],h[3],h[3],h[3]);
    else if(h[0]==15)zero_handle(h[1],h[2]);
    else if(h[0]==16)trace_product_handle(nullptr,h[1],h[2],h[3]);
    else if(h[0]==17)set_site_context(nullptr,h[1],h[2]);
    else if(h[0]==18) {
      int id=0;double scale=0;check(MPI_Bcast(&id,1,MPI_INT,0,service_comm));
      check(MPI_Bcast(&scale,1,MPI_DOUBLE,0,service_comm));
      subtract_triples(nullptr,nullptr,nullptr,nullptr,h[2],h[3],h[4],id,h[1]);
      scale_handle(id,scale);
    }
    else if(h[0]==19)solve_identity_handle(nullptr,h[1],h[3]!=0,h[2]);
    else throw std::runtime_error("unknown distributed CPU operation");
    if(trace()){std::fprintf(stderr,"service rank %d op %d end\n",service_rank,h[0]);std::fflush(stderr);}
  }
  if(service_rank==0&&std::getenv("FEDSPATIAL_DISTRIBUTED_STATS"))
    for(int op=1;op<8;++op)if(operation_calls[op])
      std::fprintf(stderr,"DISTRIBUTED_OP op=%d calls=%llu logical_bytes=%llu\n",op,
        (unsigned long long)operation_calls[op],(unsigned long long)operation_bytes[op]);
  factor_cache.reset();leader_factor_pointer=nullptr;
  pinned_rows.clear();leader_pins.clear();next_pin_id=1;
  site_context=Matrix();
  operation_bytes.fill(0);operation_calls.fill(0);
  active=false;Cblacs_gridexit(context);context=-1;service_comm=MPI_COMM_NULL;
}
bool cpu_distributed_matmul(const Matrix& x,const Matrix& y,Matrix* out) {
  if(!active)return false;
  const auto before=operation_begin();
  auto it=leader_pins.find(x.a.data());const int pin_id=it==leader_pins.end()?-1:it->second;
  command(1,x.rows,x.cols,y.cols,pin_id+1);
  *out=multiply(&x,&y,x.rows,x.cols,y.cols,pin_id);
  if(std::getenv("FEDSPATIAL_DISTRIBUTED_VERBOSE_STATS"))
    std::fprintf(stderr,"DISTRIBUTED_MATMUL m=%d k=%d n=%d pinned=%d logical_bytes=%llu\n",
      x.rows,x.cols,y.cols,pin_id>=0,
      (unsigned long long)(cpu_communication_bytes()-before));
  operation_end(1,before);return true;
}
bool cpu_distributed_matmul3(const Matrix& x,const Matrix& y,const Matrix& z,Matrix* out) {
  if(!active)return false;
  const auto before=operation_begin();
  if(trace()){std::fprintf(stderr,"service rank 0 op 4 %d %d %d %d begin\n",x.rows,x.cols,y.cols,z.cols);std::fflush(stderr);}
  auto it=leader_pins.find(x.a.data());const int pin_id=it==leader_pins.end()?-1:it->second;
  command(4,x.rows,x.cols,y.cols,z.cols);
  int pin_header=pin_id+1;check(MPI_Bcast(&pin_header,1,MPI_INT,0,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(sizeof(pin_header)*(service_size-1));
  *out=multiply3(&x,&y,&z,x.rows,x.cols,y.cols,z.cols,pin_id);
  operation_end(4,before);
  if(trace()){std::fprintf(stderr,"service rank 0 op 4 end\n");std::fflush(stderr);}return true;
}
bool cpu_distributed_subtract_triples(Matrix& initial,const Matrix* const* left,
                                      const Matrix* const* middle,
                                      const Matrix* const* right,int terms,
                                      Matrix* out) {
  if(!active)return false;
  const auto before=operation_begin();
  if(terms<=0||initial.rows!=initial.cols)throw std::runtime_error("invalid distributed triple sequence");
  const int n=initial.rows,m=left[0]->cols;
  for(int i=0;i<terms;++i)
    if(left[i]->rows!=n||left[i]->cols!=m||middle[i]->rows!=m||
       middle[i]->cols!=m||right[i]->rows!=n||right[i]->cols!=m)
      throw std::runtime_error("distributed triple sequence shape");
  command(5,n,m,terms,0);
  *out=subtract_triples(&initial,left,middle,right,n,m,terms);operation_end(5,before);return true;
}
void cpu_distributed_pin(const Matrix& matrix) {
  if(!active||leader_pins.count(matrix.a.data()))return;
  const auto before=operation_begin();
  const int id=next_pin_id++;command(6,matrix.rows,matrix.cols,id,0);
  pin_rows(&matrix,matrix.rows,matrix.cols,id);
  leader_pins[matrix.a.data()]=id;
  operation_end(6,before);
}
void cpu_distributed_unpin(const Matrix& matrix) {
  if(!active)return;
  auto it=leader_pins.find(matrix.a.data());if(it==leader_pins.end())return;
  const auto before=operation_begin();
  const int id=it->second;command(7,id,0,0,0);
  unpin_rows(id);leader_pins.erase(it);
  operation_end(7,before);
}
bool cpu_distributed_solve(const Matrix& l,const Matrix& rhs,Matrix* out) {
  if(!active)return false;
  const bool reuse=factor_cache&&factor_cache->rows==l.rows&&
                   leader_factor_pointer==l.a.data();
  command(3,l.rows,reuse?1:0,rhs.cols);
  *out=solve(&l,&rhs,l.rows,rhs.cols,reuse);return true;
}
bool cpu_distributed_cholesky(const Matrix& a,Matrix* out) {
  if(!active)return false;
  command(2,a.rows,0,0);*out=factor(&a,a.rows);
  leader_factor_pointer=out->a.data();return true;
}
bool cpu_distributed_active(){return active;}
bool cpu_distributed_trace4_symmetric(const Matrix& a,const Matrix& b,
                                      const Matrix& c,const Matrix& d,double* out) {
  if(!active)return false;
  if(!out||a.rows!=a.cols||b.rows!=a.rows||b.cols!=a.cols||
     c.rows!=a.rows||c.cols!=a.cols||d.rows!=a.rows||d.cols!=a.cols)
    throw std::runtime_error("distributed trace4 shape");
  if(a.distributed_handle<0||b.distributed_handle<0||
     c.distributed_handle<0||d.distributed_handle<0)
    throw std::runtime_error("distributed trace4 operands are not retained");
  command(8,a.distributed_handle,b.distributed_handle,
          c.distributed_handle,d.distributed_handle);
  int n=a.rows;check(MPI_Bcast(&n,1,MPI_INT,0,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(
      std::uint64_t(service_size-1)*sizeof(int));
  *out=trace4_handles(a.distributed_handle,b.distributed_handle,
                      c.distributed_handle,d.distributed_handle,n);return true;
}
bool cpu_distributed_build_subtract_triples(
    Matrix& initial,const Matrix* const* left,const Matrix* const* middle,
    const Matrix* const* right,int terms,int* handle) {
  if(!active)return false;
  if(!handle||initial.rows!=initial.cols||terms<0)
    throw std::runtime_error("invalid distributed derivative build");
  const int n=initial.rows,m=terms?left[0]->cols:1,id=next_pin_id++;
  for(int i=0;i<terms;++i)
    if(left[i]->rows!=n||left[i]->cols!=m||middle[i]->rows!=m||
       middle[i]->cols!=m||right[i]->rows!=n||right[i]->cols!=m)
      throw std::runtime_error("distributed derivative build shape");
  command(9,n,m,terms,id);
  subtract_triples(&initial,left,middle,right,n,m,terms,id);
  *handle=id;return true;
}
bool cpu_distributed_matmul_handle(int handle,const Matrix& rhs,Matrix* out) {
  if(!active)return false;
  auto it=pinned_rows.find(handle);if(it==pinned_rows.end()||rhs.rows!=it->second.cols)
    throw std::runtime_error("distributed handle matmul shape");
  const int n=rhs.rows;command(10,handle,n,rhs.cols,0);
  *out=multiply(nullptr,&rhs,n,n,rhs.cols,handle);return true;
}
bool cpu_distributed_trace_product_handle(const Matrix& matrix,int handle,double* out) {
  if(!active)return false;
  if(!out||matrix.rows!=matrix.cols)throw std::runtime_error("distributed handle trace shape");
  auto pinned=leader_pins.find(matrix.a.data());const int matrix_id=pinned==leader_pins.end()?-1:pinned->second;
  command(11,matrix_id,handle,matrix.rows,0);
  *out=trace_product_handle(&matrix,matrix_id,handle,matrix.rows);return true;
}
void cpu_distributed_scale_handle(int handle,double scale) {
  if(!active)return;command(12,handle,0,0,0);
  check(MPI_Bcast(&scale,1,MPI_DOUBLE,0,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(
      std::uint64_t(service_size-1)*sizeof(double));scale_handle(handle,scale);
}
void cpu_distributed_unpin_handle(int handle) {
  if(!active)return;command(7,handle,0,0,0);unpin_rows(handle);
}
int cpu_distributed_release_to_handle(Matrix& matrix) {
  if(!active)return -1;
  auto it=leader_pins.find(matrix.a.data());if(it==leader_pins.end())
    throw std::runtime_error("matrix must be pinned before release");
  const int id=it->second;leader_pins.erase(it);
  matrix.a.clear();matrix.a.shrink_to_fit();matrix.distributed_handle=id;return id;
}
int cpu_distributed_clone_scale_handle(int source,double scale) {
  if(!active)return -1;const int id=next_pin_id++;
  command(13,source,id,0,0);check(MPI_Bcast(&scale,1,MPI_DOUBLE,0,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(
      std::uint64_t(service_size-1)*sizeof(double));
  return clone_scale_handle(source,id,scale);
}
int cpu_distributed_zero_handle(int n) {
  if(!active)return -1;const int id=next_pin_id++;command(15,id,n,0,0);
  return zero_handle(id,n);
}
int cpu_distributed_inverse_handle(const Matrix& l) {
  if(!active)return -1;
  const bool reuse=factor_cache&&factor_cache->rows==l.rows&&
                   leader_factor_pointer==l.a.data();
  const int id=next_pin_id++;command(19,l.rows,id,reuse?1:0,0);
  return solve_identity_handle(&l,l.rows,reuse,id);
}
bool cpu_distributed_matmul_handles(int left,int right,int m,int k,int n,Matrix* out) {
  if(!active)return false;
  if(!out||m!=k||k!=n)throw std::runtime_error("distributed handle product requires square operands");
  command(14,left,right,n,0);*out=multiply_handles(left,right,m,k,n);return true;
}
bool cpu_distributed_trace_product_handles(int left,int right,double* out) {
  if(!active)return false;
  auto it=pinned_rows.find(left);if(!out||it==pinned_rows.end())
    throw std::runtime_error("invalid distributed handle trace");
  const int n=it->second.cols;command(16,left,right,n,0);
  *out=trace_product_handle(nullptr,left,right,n);return true;
}
void cpu_distributed_set_site_context(const Matrix& sites,int nu2,double alpha,double range) {
  if(!active)return;
  if(sites.cols!=2||(nu2!=1&&nu2!=3&&nu2!=5)||!(alpha>0)||!(range>0))
    throw std::runtime_error("invalid distributed site context");
  command(17,sites.rows,nu2,0,0);set_site_context(&sites,sites.rows,nu2,alpha,range);
}
bool cpu_distributed_build_site_subtract_triples(
    int kind,int n,const Matrix* const* left,const Matrix* const* middle,
    const Matrix* const* right,int terms,double scale,int* handle) {
  if(!active)return false;
  if(!handle||kind<0||kind>4||terms<0)throw std::runtime_error("invalid distributed site derivative build");
  const int m=terms?left[0]->cols:1;int id=next_pin_id++;
  for(int i=0;i<terms;++i)
    if(left[i]->rows!=n||left[i]->cols!=m||middle[i]->rows!=m||
       middle[i]->cols!=m||right[i]->rows!=n||right[i]->cols!=m)
      throw std::runtime_error("distributed site derivative shape");
  command(18,kind,n,m,terms);check(MPI_Bcast(&id,1,MPI_INT,0,service_comm));
  check(MPI_Bcast(&scale,1,MPI_DOUBLE,0,service_comm));
  if(service_rank==0)cpu_add_communication_bytes(
      std::uint64_t(service_size-1)*(sizeof(int)+sizeof(double)));
  subtract_triples(nullptr,left,middle,right,n,m,terms,id,kind);scale_handle(id,scale);
  *handle=id;return true;
}
}
