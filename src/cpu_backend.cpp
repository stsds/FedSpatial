#include "fedspatial/cpu_backend.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <initializer_list>
namespace fedspatial {
namespace {
void axpy(Matrix& a, const Matrix& b, double scale = 1.0) {
  if(a.rows!=b.rows||a.cols!=b.cols)throw std::runtime_error("CPU axpy shape");
  for (int i = 0; i < a.rows * a.cols; ++i) a.a[i] += scale * b.a[i];
}

Matrix scaled(Matrix a, double scale) {
  for (double& value : a.a) value *= scale;
  return a;
}

double trace(const Matrix& a) {
  double value = 0.0;
  for (int i = 0; i < a.rows; ++i) value += a(i, i);
  return value;
}

double inner(const Matrix& a, const Matrix& b) {
  double value = 0.0;
  for (int i = 0; i < a.rows * a.cols; ++i) value += a.a[i] * b.a[i];
  return value;
}

Matrix right_solve_spd(const Matrix& a, const Matrix& k) {
  return transpose(solve_spd(k, transpose(a)));
}

Matrix zero(int rows, int cols) { return Matrix(rows, cols); }

struct TripleTerm { const Matrix* left; const Matrix* middle; const Matrix* right; };
void subtract_triples(Matrix& out,std::initializer_list<TripleTerm> terms) {
#ifdef FEDSPATIAL_SCALAPACK
  std::vector<const Matrix*> left,middle,right;
  left.reserve(terms.size());middle.reserve(terms.size());right.reserve(terms.size());
  for(const auto& term:terms) {
    left.push_back(term.left);middle.push_back(term.middle);right.push_back(term.right);
  }
  Matrix distributed;
  if(cpu_distributed_subtract_triples(out,left.data(),middle.data(),right.data(),
                                      int(terms.size()),&distributed)) {
    out=std::move(distributed);return;
  }
#endif
  for(const auto& term:terms)
    axpy(out,matmul3(*term.left,*term.middle,transpose(*term.right)),-1.0);
}
Matrix retain_subtract_triples(Matrix initial,std::initializer_list<TripleTerm> terms,
                               double scale=1.0) {
#ifdef FEDSPATIAL_SCALAPACK
  std::vector<const Matrix*> left,middle,right;
  left.reserve(terms.size());middle.reserve(terms.size());right.reserve(terms.size());
  for(const auto& term:terms) {
    left.push_back(term.left);middle.push_back(term.middle);right.push_back(term.right);
  }
  int handle=-1;
  if(cpu_distributed_build_subtract_triples(
       initial,left.data(),middle.data(),right.data(),int(terms.size()),&handle)) {
    if(scale!=1.0)cpu_distributed_scale_handle(handle,scale);
    Matrix result;result.rows=initial.rows;result.cols=initial.cols;
    result.distributed_handle=handle;return result;
  }
#endif
  subtract_triples(initial,terms);return scaled(std::move(initial),scale);
}
Matrix retain_site_subtract_triples(int kind,int n,
                                    std::initializer_list<TripleTerm> terms,
                                    double scale=1.0) {
#ifdef FEDSPATIAL_SCALAPACK
  std::vector<const Matrix*> left,middle,right;
  left.reserve(terms.size());middle.reserve(terms.size());right.reserve(terms.size());
  for(const auto& term:terms) {
    left.push_back(term.left);middle.push_back(term.middle);right.push_back(term.right);
  }
  int handle=-1;
  if(cpu_distributed_build_site_subtract_triples(
       kind,n,left.data(),middle.data(),right.data(),int(terms.size()),scale,&handle)) {
    Matrix result;result.rows=n;result.cols=n;result.distributed_handle=handle;return result;
  }
#endif
  throw std::runtime_error("distributed site derivative build unavailable");
}

Matrix covariance_first(const Matrix& x, const Matrix& y,
                        const GlobalParameters& parameters, int parameter) {
  const auto derivatives = cpu_matern(
      x, y, parameters.nu2, parameters.alpha, parameters.range);
  return parameter == 1 ? derivatives.alpha : derivatives.range;
}

// Covariance-only counterpart to cpu_matern.  The local-state builders never
// consume derivative fields, so allocating and evaluating seven dense panels
// there multiplied both memory traffic and peak storage without changing the
// model.  Keep the value expression and FP64 operation order identical to the
// value field of cpu_matern.
Matrix covariance_value(const Matrix& x,const Matrix& y,int nu2,
                        double alpha,double range) {
  if(x.rows<=0||y.rows<=0||x.a.size()!=size_t(x.rows)*size_t(x.cols)||
     y.a.size()!=size_t(y.rows)*size_t(y.cols)||x.cols!=2||y.cols!=2||
     (nu2!=1&&nu2!=3&&nu2!=5)||!(alpha>0&&range>0)||
     !std::isfinite(alpha)||!std::isfinite(range))
    throw std::runtime_error("invalid CPU Matern arguments");
  Matrix value(x.rows,y.rows);
  const int begin=cpu_row_begin(x.rows),end=cpu_row_end(x.rows);
#pragma omp parallel for schedule(static)
  for(int i=begin;i<end;++i) for(int j=0;j<y.rows;++j) {
    double dx=x(i,0)-y(j,0),dy=x(i,1)-y(j,1);
    double u=std::sqrt(dx*dx+dy*dy)/range,e,p;
    if(nu2==1){e=std::exp(-u);p=e;}
    else if(nu2==3){u*=std::sqrt(3.);e=std::exp(-u);p=(1+u)*e;}
    else{u*=std::sqrt(5.);e=std::exp(-u);p=(1+u+u*u/3)*e;}
    value(i,j)=alpha*p;
  }
  cpu_gather_rows(value);
  return value;
}

// Apply a dense operator chain to the thin Q factor from right to left.
// No n-by-n chain product is formed; worker-local panel DGEMM is unchanged.
using MatrixChain = std::initializer_list<const Matrix*>;
double q_contract(MatrixChain chain, const Matrix& l, const Matrix& r,
                  const Matrix& sigma) {
  Matrix action = matmul(l, sigma);
  for (auto it = chain.end(); it != chain.begin();) {
    --it;
    action = matmul(**it, action);
  }
  return inner(r, action);
}

double trace_product(const Matrix& a, const Matrix& b) {
#ifdef FEDSPATIAL_SCALAPACK
  if(a.distributed_handle>=0&&b.distributed_handle>=0) {
    double value=0;
    if(cpu_distributed_trace_product_handles(a.distributed_handle,
                                             b.distributed_handle,&value))return value;
    throw std::runtime_error("distributed trace product outside worker service");
  }
  if(b.distributed_handle>=0) {
    double value=0;
    if(cpu_distributed_trace_product_handle(a,b.distributed_handle,&value))return value;
    throw std::runtime_error("distributed trace product outside worker service");
  }
#endif
  if (a.cols != b.rows || a.rows != b.cols)
    throw std::runtime_error("CPU trace product shape");
  // Stream rows of the contraction instead of allocating a.rows-by-b.cols.
  double value = 0.0;
  for (int i = 0; i < a.rows; ++i) {
    double diagonal = 0.0;
    for (int k = 0; k < a.cols; ++k) diagonal += a(i, k) * b(k, i);
    value += diagonal;
  }
  return value;
}

// Internal builder borrows the fixed state for the same worker and parameters.
PDerivativeWorkspace cpu_derivatives_from_local(
    const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters, const LocalComputationResult& local);

// Production-only workspace: thin covariance derivatives, three M first
// derivatives, and one second derivative owned by the current Hessian pair.
struct StreamedMDerivatives {
  const WorkerInput& worker;
  const GlobalParameters& p;
  const LocalComputationResult& local;
  MaternDerivatives kd, cd;
  Matrix kl, ba, br;
  BSecondDerivatives b_second;
  MFirstDerivatives m_first;
  std::array<Matrix,3> site_terms;

  Matrix rs(const Matrix& x) const {
    return transpose(solve_cholesky(kl, transpose(x)));
  }
  // Evaluate site covariance primitives in bounded row tiles. Retain the two
  // first derivatives and alpha-range. Range-range is generated only after
  // alpha-range has been consumed, so both Hessian site terms never coexist.
  void build_site_terms() {
    const int n = worker.sites.rows;
    for(auto& term:site_terms) term=Matrix(n,n);
    for (int begin=0; begin<n; begin+=32) {
      const int rows=std::min(32,n-begin);
      Matrix sites(rows,2);
      for(int i=0;i<rows;++i) for(int j=0;j<2;++j)
        sites(i,j)=worker.sites(begin+i,j);
      auto tile=cpu_matern(sites,worker.sites,p.nu2,p.alpha,p.range);
      const Matrix* terms[]={&tile.alpha,&tile.range,&tile.ar};
      for(int t=0;t<3;++t) for(int i=0;i<rows;++i) for(int j=0;j<n;++j)
        site_terms[t](begin+i,j)=(*terms[t])(i,j);
    }
  }
  Matrix build_site_rr() const {
    const int n=worker.sites.rows;Matrix rr(n,n);
    const int begin=cpu_row_begin(n),end=cpu_row_end(n);
#pragma omp parallel for schedule(static)
    for(int i=begin;i<end;++i)for(int j=0;j<n;++j) {
      double dx=worker.sites(i,0)-worker.sites(j,0);
      double dy=worker.sites(i,1)-worker.sites(j,1);
      double u=std::sqrt(dx*dx+dy*dy)/p.range,e,rbb;
      if(p.nu2==1){e=std::exp(-u);rbb=p.alpha*e*(u*u-2*u)/(p.range*p.range);}
      else if(p.nu2==3){u*=std::sqrt(3.);e=std::exp(-u);rbb=p.alpha*e*(u*u*u-3*u*u)/(p.range*p.range);}
      else{u*=std::sqrt(5.);e=std::exp(-u);rbb=p.alpha*e*(u*u*u*u-3*u*u*u-3*u*u)/(3*p.range*p.range);}
      rr(i,j)=rbb;
    }
    cpu_gather_rows(rr);
    return rr;
  }
  StreamedMDerivatives(const WorkerInput& w,const Matrix& knots,
                      const GlobalParameters& parameters,LocalComputationResult& state)
      : worker(w),p(parameters),local(state),
        kd(cpu_matern(knots,knots,p.nu2,p.alpha,p.range)),
        cd(cpu_matern(w.sites,knots,p.nu2,p.alpha,p.range)),
        kl(cholesky_lower(kd.value)) {
    ba=rs(cd.alpha);axpy(ba,rs(matmul(local.B,kd.alpha)),-1);
    br=rs(cd.range);axpy(br,rs(matmul(local.B,kd.range)),-1);
    auto b2=[&](const Matrix& cij,const Matrix& ci,const Matrix& kj,const Matrix& bj,const Matrix& ki,const Matrix& kij){
      Matrix out=rs(cij);axpy(out,rs(matmul(rs(ci),kj)),-1);
      axpy(out,rs(matmul(bj,ki)),-1);axpy(out,rs(matmul(local.B,kij)),-1);
      axpy(out,rs(matmul(rs(matmul(local.B,ki)),kj)));return out;
    };
    b_second={kd.aa,kd.ar,kd.rr,cd.aa,cd.ar,cd.rr,
      b2(cd.aa,cd.alpha,kd.alpha,ba,kd.alpha,kd.aa),
      b2(cd.ar,cd.alpha,kd.range,br,kd.alpha,kd.ar),
      b2(cd.ra,cd.range,kd.alpha,ba,kd.range,kd.ra),
      b2(cd.rr,cd.range,kd.range,br,kd.range,kd.rr)};
    auto a1=[&](Matrix out,const Matrix& bi,const Matrix& ki){
      subtract_triples(out,{{&bi,&kd.value,&local.B},
                            {&local.B,&ki,&local.B},
                            {&local.B,&kd.value,&bi}});return out;
    };
    // Block 3 is the final consumer of M in a synchronous round. Transfer its
    // storage into the delta derivative instead of retaining two n-by-n copies.
    Matrix av=std::move(state.M);add_diagonal(av,-1);av=scaled(std::move(av),1./p.delta);
    m_first.delta=std::move(av);
#ifdef FEDSPATIAL_SCALAPACK
    if(cpu_distributed_active()) {
      cpu_distributed_set_site_context(worker.sites,p.nu2,p.alpha,p.range);
      m_first.alpha=retain_site_subtract_triples(1,worker.sites.rows,
        {{&ba,&kd.value,&local.B},{&local.B,&kd.alpha,&local.B},
         {&local.B,&kd.value,&ba}},p.delta);
      m_first.range=retain_site_subtract_triples(2,worker.sites.rows,
        {{&br,&kd.value,&local.B},{&local.B,&kd.range,&local.B},
         {&local.B,&kd.value,&br}},p.delta);
    } else
#endif
    {
      build_site_terms();
      m_first.alpha=scaled(a1(std::move(site_terms[0]),ba,kd.alpha),p.delta);
      m_first.range=scaled(a1(std::move(site_terms[1]),br,kd.range),p.delta);
    }
  }
  Matrix second(int i,int j) {
    if(i==0) {
      if(j==0) return Matrix(worker.sites.rows,worker.sites.rows);
      return scaled(j==1?m_first.alpha:m_first.range,1./p.delta);
    }
    const Matrix& bi=i==1?ba:br;
    const Matrix& bj=j==1?ba:br;
    const Matrix& ki=i==1?kd.alpha:kd.range;
    const Matrix& kj=j==1?kd.alpha:kd.range;
    const Matrix& bij=j==1?b_second.baa:(i==1?b_second.bar:b_second.brr);
    const Matrix& kij=j==1?kd.aa:(i==1?kd.ar:kd.rr);
    Matrix out=j==1?Matrix(worker.sites.rows,worker.sites.rows):
                      (i==1?std::move(site_terms[2]):build_site_rr());
    subtract_triples(out,{{&bij,&kd.value,&local.B},{&bi,&kj,&local.B},
      {&bi,&kd.value,&bj},{&bj,&ki,&local.B},{&local.B,&kij,&local.B},
      {&local.B,&ki,&bj},{&bj,&kd.value,&bi},{&local.B,&kj,&bi},
      {&local.B,&kd.value,&bij}});
    return scaled(std::move(out),p.delta);
  }
#ifdef FEDSPATIAL_SCALAPACK
  Matrix second_distributed(int i,int j) {
    if(i==0) {
      Matrix result;result.rows=worker.sites.rows;result.cols=worker.sites.rows;
      result.distributed_handle=j==0?cpu_distributed_zero_handle(worker.sites.rows):
        cpu_distributed_clone_scale_handle(
          (j==1?m_first.alpha:m_first.range).distributed_handle,1./p.delta);
      return result;
    }
    const Matrix& bi=i==1?ba:br;
    const Matrix& bj=j==1?ba:br;
    const Matrix& ki=i==1?kd.alpha:kd.range;
    const Matrix& kj=j==1?kd.alpha:kd.range;
    const Matrix& bij=j==1?b_second.baa:(i==1?b_second.bar:b_second.brr);
    const Matrix& kij=j==1?kd.aa:(i==1?kd.ar:kd.rr);
    const int site_kind=j==1?0:(i==1?3:4);
    return retain_site_subtract_triples(site_kind,worker.sites.rows,
      {{&bij,&kd.value,&local.B},{&bi,&kj,&local.B},{&bi,&kd.value,&bj},
       {&bj,&ki,&local.B},{&local.B,&kij,&local.B},{&local.B,&ki,&bj},
       {&bj,&kd.value,&bi},{&local.B,&kj,&bi},{&local.B,&kd.value,&bij}},p.delta);
  }
#endif
};

struct Eigen3 { Matrix vectors; std::array<double, 3> values; };
Eigen3 symmetric_eigen3(Matrix matrix) {
  Matrix vectors = identity(3);
  for (int iteration = 0; iteration < 64; ++iteration) {
    int p = 0, q = 1; double maximum = std::abs(matrix(0, 1));
    for (int i = 0; i < 3; ++i) for (int j = i + 1; j < 3; ++j)
      if (std::abs(matrix(i, j)) > maximum) { maximum = std::abs(matrix(i, j)); p = i; q = j; }
    if (maximum < 1e-15) break;
    const double angle = .5 * std::atan2(2. * matrix(p, q), matrix(q, q) - matrix(p, p));
    const double c = std::cos(angle), s = std::sin(angle);
    for (int k = 0; k < 3; ++k) { const double pk = matrix(p, k), qk = matrix(q, k); matrix(p, k) = c * pk - s * qk; matrix(q, k) = s * pk + c * qk; }
    for (int k = 0; k < 3; ++k) { const double kp = matrix(k, p), kq = matrix(k, q); matrix(k, p) = c * kp - s * kq; matrix(k, q) = s * kp + c * kq; const double vp = vectors(k, p), vq = vectors(k, q); vectors(k, p) = c * vp - s * vq; vectors(k, q) = s * vp + c * vq; }
  }
  std::array<int, 3> order{0, 1, 2}; std::array<double, 3> raw{matrix(0,0), matrix(1,1), matrix(2,2)};
  std::sort(order.begin(), order.end(), [&](int i, int j) { return raw[i] < raw[j]; });
  Matrix sorted_vectors(3, 3); std::array<double, 3> sorted_values{};
  for (int column = 0; column < 3; ++column) { sorted_values[column] = raw[order[column]]; for (int row = 0; row < 3; ++row) sorted_vectors(row, column) = vectors(row, order[column]); }
  return {sorted_vectors, sorted_values};
}
}
DeltaThetaLocalResult cpu_delta_theta_from_local(
    const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters,
    LocalComputationResult& local, bool fixed_zero_mean) {
  const int n = worker.sites.rows;
  const int m = knots.rows;
  const Matrix covariance_k = covariance_value(
      knots, knots, parameters.nu2, parameters.alpha, parameters.range);
  // r has Python's sign: X gamma - z + B mu.  The sign is immaterial to
  // the quadratic value but is material when differentiating r.
  Matrix r = matmul(local.B, parameters.mu);
  // The regression path must use the newly updated gamma, whereas the
  // benchmark-only fixed-zero-mean path reuses z verbatim and never enters
  // an X/gamma kernel.
  const Matrix current_e=fixed_zero_mean
      ?local.e:subtract(worker.z,matmul(worker.x,parameters.gamma));
  axpy(r, current_e, -1.0);
  // Reuse the worker factor for all fixed-state solves.
  Matrix w;
#ifdef FEDSPATIAL_SCALAPACK
  if(cpu_distributed_active()) {
    w.rows=n;w.cols=n;
    w.distributed_handle=cpu_distributed_inverse_handle(local.chol_M);
  } else
#endif
  w = solve_cholesky(local.chol_M, identity(n));
  const Matrix u = solve_cholesky(local.chol_M, r);
  auto tq = [&](MatrixChain a) {
    return q_contract(a, local.B, local.B, parameters.sigma);
  };
  const double quadratic = inner(r, u);
  const double trace_q = tq({&w});

  // Build derivatives from the existing fixed state without another local workspace.
  StreamedMDerivatives derivatives(worker, knots, parameters, local);
  const auto& b_second = derivatives.b_second;
  auto& m_first = derivatives.m_first;

  std::array<Matrix, 3> b_first{zero(n, m), zero(n, m), zero(n, m)};
  for (int parameter = 1; parameter < 3; ++parameter) {
    const Matrix c_i = covariance_first( worker.sites, knots,
                                        parameters, parameter);
    const Matrix k_i = covariance_first( knots, knots, parameters,
                                        parameter);
    b_first[parameter] = right_solve_spd(c_i, covariance_k);
    axpy(b_first[parameter],
         right_solve_spd(matmul(local.B, k_i), covariance_k), -1.0);
  }

  std::array<Matrix, 3> r_first{zero(n, 1), zero(n, 1), zero(n, 1)};

  for (int parameter = 1; parameter < 3; ++parameter) {
    r_first[parameter] = matmul(b_first[parameter], parameters.mu);
  }

  // Borrow dense M derivatives: copying these arrays costs twelve n-by-n buffers.
  Matrix* m1[3] = {&m_first.delta, &m_first.alpha, &m_first.range};
#ifdef FEDSPATIAL_SCALAPACK
  // These four dense operands are reused throughout every gradient/Hessian
  // contraction. Keep their owned row panels resident on the service ranks.
  if(w.distributed_handle<0)cpu_distributed_pin(w);
  for(Matrix* derivative:m1)if(derivative->distributed_handle<0)
    cpu_distributed_pin(*derivative);
  if(w.distributed_handle<0)cpu_distributed_release_to_handle(w);
  for(Matrix* derivative:m1)if(derivative->distributed_handle<0)
    cpu_distributed_release_to_handle(*derivative);
#endif


  std::array<std::array<Matrix, 3>, 3> b2{{
      {zero(n, m), zero(n, m), zero(n, m)},
      {zero(n, m), b_second.baa, b_second.bar},
      {zero(n, m), b_second.bra, b_second.brr},
  }};

  auto tq1 = [&](MatrixChain a, int i) {
    if (i == 0) return 0.0;
    return q_contract(a, b_first[i], local.B, parameters.sigma) +
           q_contract(a, local.B, b_first[i], parameters.sigma);
  };
  auto tq2 = [&](MatrixChain a, int i, int j) {
    if (i == 0 || j == 0) return 0.0;
    return q_contract(a, b2[i][j], local.B, parameters.sigma) +
           q_contract(a, local.B, b2[i][j], parameters.sigma) +
           q_contract(a, b_first[i], b_first[j], parameters.sigma) +
           q_contract(a, b_first[j], b_first[i], parameters.sigma);
  };

  DeltaThetaLocalResult out;
  out.objective = -n * std::log(parameters.delta) +
                  std::log(parameters.delta) * n + local.logdet_R +
                  parameters.delta * (quadratic + trace_q);
  // local.logdet_R = log|M| - n log(delta), so the expression above is log|M|.

  std::array<double, 3> h{};
  std::array<double, 3> h_first{};
  for (int i = 0; i < 3; ++i) {
    h[i] = quadratic + trace_q;
    const double q_i = 2.0 * inner(r_first[i], u) -
                       inner(u, matmul(*m1[i], u));
    const double t_i = tq1({&w}, i) -
                       tq({&w, m1[i], &w});
    h_first[i] = q_i + t_i;
    out.gradient[i] = trace_product(w, *m1[i]) +
                      (i == 0 ? -static_cast<double>(n) / parameters.delta : 0.0) +
                      (i == 0 ? h[i] : 0.0) + parameters.delta * h_first[i];
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      Matrix m_ij;
#ifdef FEDSPATIAL_SCALAPACK
      if(cpu_distributed_active())m_ij=derivatives.second_distributed(i,j);
      else
#endif
      m_ij=derivatives.second(i,j);
      Matrix r_ij = matmul(b2[i][j], parameters.mu);

      const Matrix w_rj = matmul(w, r_first[j]);
      Matrix w_mj_u,w_mi_u;
#ifdef FEDSPATIAL_SCALAPACK
      if(cpu_distributed_active()) {
        w_mj_u=matmul(w,matmul(*m1[j],u));
        w_mi_u=matmul(w,matmul(*m1[i],u));
      } else
#endif
      {
        w_mj_u=matmul3(w,*m1[j],u);
        w_mi_u=matmul3(w,*m1[i],u);
      }
      double dense_quadratic_terms;
#ifdef FEDSPATIAL_SCALAPACK
      if(cpu_distributed_active())
        dense_quadratic_terms =
          inner(u,matmul(*m1[j],matmul(w,matmul(*m1[i],u)))) +
          inner(u,matmul(*m1[i],matmul(w,matmul(*m1[j],u))));
      else
#endif
        dense_quadratic_terms =
          inner(u, matmul(matmul(*m1[j], w), matmul(*m1[i], u))) +
          inner(u, matmul(matmul(*m1[i], w), matmul(*m1[j], u)));
      const double quadratic_ij =
          2.0 * inner(r_ij, u) + 2.0 * inner(r_first[i], w_rj) -
          2.0 * inner(r_first[i], w_mj_u) -
          2.0 * inner(r_first[j], w_mi_u) +
          dense_quadratic_terms -
          inner(u, matmul(m_ij, u));

      const double trace_ij =
          tq2({&w}, i, j) -
          tq1({&w, m1[j], &w}, i) -
          tq1({&w, m1[i], &w}, j) +
          tq({&w, m1[j], &w, m1[i], &w}) +
          tq({&w, m1[i], &w, m1[j], &w}) -
          tq({&w, &m_ij, &w});

      double dense_trace_product;
#ifdef FEDSPATIAL_SCALAPACK
      if(cpu_distributed_active()) {
        if(!cpu_distributed_trace4_symmetric(w,*m1[j],w,*m1[i],
                                             &dense_trace_product))
          throw std::runtime_error("distributed trace4 unavailable");
      } else
#endif
        dense_trace_product=trace_product(matmul(w,*m1[j]),matmul(w,*m1[i]));
      const double value =
          (i == 0 && j == 0
               ? static_cast<double>(n) / (parameters.delta * parameters.delta)
               : 0.0) +
          trace_product(w, m_ij) -
          dense_trace_product +
          (j == 0 ? h_first[i] : 0.0) +
          (i == 0 ? h_first[j] : 0.0) +
          parameters.delta * (quadratic_ij + trace_ij);
      out.hessian[3 * i + j] = value;
      out.hessian[3 * j + i] = value;
#ifdef FEDSPATIAL_SCALAPACK
      if(m_ij.distributed_handle>=0)
        cpu_distributed_unpin_handle(m_ij.distributed_handle);
#endif
    }
  }
#ifdef FEDSPATIAL_SCALAPACK
  for(Matrix* derivative:m1)cpu_distributed_unpin_handle(derivative->distributed_handle);
  cpu_distributed_unpin_handle(w.distributed_handle);
#endif
  return out;
}
DeltaThetaLocalResult cpu_delta_theta(
    const WorkerInput& worker, const Matrix& knots,
    const GlobalParameters& parameters) {
  auto local=cpu_local(worker,knots,parameters);
  return cpu_delta_theta_from_local(worker,knots,parameters,local);
}
void cpu_k_correction(const Matrix& knots, const GlobalParameters& p,
                         Matrix* gradient, Matrix* hessian) {
  if(!gradient||!hessian)throw std::runtime_error("null CPU K correction output");
  *gradient=Matrix(3,1);*hessian=Matrix(3,3);
  const auto d = cpu_matern( knots, knots, p.nu2, p.alpha, p.range);
  Matrix first[3] = {Matrix(knots.rows, knots.rows), d.alpha, d.range};
  Matrix second[3][3] = {{Matrix(knots.rows, knots.rows), Matrix(knots.rows, knots.rows), Matrix(knots.rows, knots.rows)},
                          {Matrix(knots.rows, knots.rows), d.aa, d.ar},
                          {Matrix(knots.rows, knots.rows), d.ra, d.rr}};
  const Matrix inverse = solve_spd(d.value, identity(knots.rows));
  const Matrix inverse_mu = matmul(inverse, p.mu);
  for (int i = 1; i < 3; ++i) {
    (*gradient)(i, 0) = trace(matmul(inverse, first[i])) - inner(inverse_mu, matmul(first[i], inverse_mu)) - trace(matmul(matmul(matmul(inverse, first[i]), inverse), p.sigma));
    for (int j = 1; j < 3; ++j) {
      (*hessian)(i, j) = trace(matmul(inverse, second[i][j])) - trace(matmul(matmul(matmul(inverse, first[j]), inverse), first[i])) - inner(inverse_mu, matmul(second[i][j], inverse_mu)) + inner(inverse_mu, matmul(matmul(first[j], inverse), matmul(first[i], inverse_mu))) + inner(inverse_mu, matmul(matmul(first[i], inverse), matmul(first[j], inverse_mu))) + trace(matmul(matmul(matmul(matmul(inverse, first[j]), inverse), first[i]), matmul(inverse, p.sigma))) + trace(matmul(matmul(matmul(matmul(inverse, first[i]), inverse), first[j]), matmul(inverse, p.sigma))) - trace(matmul(matmul(matmul(inverse, second[i][j]), inverse), p.sigma));
    }
  }
}

MaternDerivatives cpu_matern(const Matrix& x,const Matrix& y,int nu2,double alpha,double range) {
  if(x.rows<=0||y.rows<=0||x.a.size()!=size_t(x.rows)*size_t(x.cols)||y.a.size()!=size_t(y.rows)*size_t(y.cols)||x.cols!=2||y.cols!=2||(nu2!=1&&nu2!=3&&nu2!=5)||
     !(alpha>0&&range>0)||!std::isfinite(alpha)||!std::isfinite(range))
    throw std::runtime_error("invalid CPU Matern arguments");
  MaternDerivatives o{Matrix(x.rows,y.rows),Matrix(x.rows,y.rows),Matrix(x.rows,y.rows),
                     Matrix(x.rows,y.rows),Matrix(x.rows,y.rows),Matrix(x.rows,y.rows),Matrix(x.rows,y.rows)};
  const int begin=cpu_row_begin(x.rows),end=cpu_row_end(x.rows);
#pragma omp parallel for schedule(static)
  for(int i=begin;i<end;++i) for(int j=0;j<y.rows;++j) {
    double dx=x(i,0)-y(j,0),dy=x(i,1)-y(j,1),u=std::sqrt(dx*dx+dy*dy)/range,e,p,rb,rbb;
    if(nu2==1){e=std::exp(-u);p=e;rb=e*u/range;rbb=alpha*e*(u*u-2*u)/(range*range);}
    else if(nu2==3){u*=std::sqrt(3.);e=std::exp(-u);p=(1+u)*e;rb=e*u*u/range;rbb=alpha*e*(u*u*u-3*u*u)/(range*range);}
    else{u*=std::sqrt(5.);e=std::exp(-u);p=(1+u+u*u/3)*e;rb=e*(u*u+u*u*u)/(3*range);rbb=alpha*e*(u*u*u*u-3*u*u*u-3*u*u)/(3*range*range);}
    o.value(i,j)=alpha*p;o.alpha(i,j)=p;o.range(i,j)=alpha*rb;
    o.ar(i,j)=o.ra(i,j)=rb;o.rr(i,j)=rbb;
  }
  for(Matrix* a:{&o.value,&o.alpha,&o.range,&o.aa,&o.ar,&o.ra,&o.rr})cpu_gather_rows(*a);
  return o;
}

LocalComputationResult cpu_local(const WorkerInput& w,const Matrix& knots,
                                 const GlobalParameters& p,bool fixed_zero_mean) {
  const int n=w.sites.rows,m=knots.rows;
  if(n<=0||m<=0||w.x.rows!=n||w.z.rows!=n||w.z.cols!=1||
     p.gamma.rows!=w.x.cols||p.gamma.cols!=1||p.mu.rows!=m||p.mu.cols!=1||
     p.sigma.rows!=m||p.sigma.cols!=m||!(p.delta>0)||!std::isfinite(p.delta))
    throw std::runtime_error("invalid CPU worker/model dimensions or delta");
  LocalComputationResult o;
  auto k=covariance_value(knots,knots,p.nu2,p.alpha,p.range);
  o.cstar=covariance_value(w.sites,knots,p.nu2,p.alpha,p.range);
  o.B=right_solve_spd(o.cstar,k);
  o.M=subtract(covariance_value(w.sites,w.sites,p.nu2,p.alpha,p.range),
               matmul(matmul(o.B,k),transpose(o.B)));
  o.M=scaled(o.M,p.delta);add_diagonal(o.M,1.);
  o.chol_M=cholesky_lower(o.M);
  o.e=fixed_zero_mean?w.z:subtract(w.z,matmul(w.x,p.gamma));
  o.rinv_B=solve_cholesky(o.chol_M,o.B);o.rinv_e=solve_cholesky(o.chol_M,o.e);
  o.bt_rinv_B=matmul(transpose(o.B),o.rinv_B);
  o.bt_rinv_e=matmul(transpose(o.B),o.rinv_e);
  o.summary={scaled(o.bt_rinv_e,-1.),o.bt_rinv_B};
  for(int i=0;i<n;++i)o.logdet_R+=2*std::log(o.chol_M(i,i));
  o.logdet_R-=n*std::log(p.delta);
  o.et_rinv_e=p.delta*inner(o.e,o.rinv_e);
  auto r=subtract(matmul(o.B,p.mu),o.e);
  o.local_objective=o.logdet_R+p.delta*(inner(r,solve_cholesky(o.chol_M,r))+trace_product(o.bt_rinv_B,p.sigma));
  // These compatibility fields follow the existing R^-1 = delta M^-1 convention.
  o.rinv_B=scaled(o.rinv_B,p.delta);o.rinv_e=scaled(o.rinv_e,p.delta);
  o.bt_rinv_B=scaled(o.bt_rinv_B,p.delta);o.bt_rinv_e=scaled(o.bt_rinv_e,p.delta);
  return o;
}
GammaLocalSummary cpu_gamma(const WorkerInput& w,const Matrix& knots,const GlobalParameters& p) {
  const auto o=cpu_local(w,knots,p);
  return cpu_gamma_from_local(w,p,o);
}
GammaLocalSummary cpu_gamma_from_local(const WorkerInput& w,
                                       const GlobalParameters& p,
                                       const LocalComputationResult& o) {
  auto rhs=subtract(w.z,matmul(o.B,p.mu));
  return {matmul(transpose(w.x),solve_cholesky(o.chol_M,rhs)),
          matmul(transpose(w.x),solve_cholesky(o.chol_M,w.x))};
}

namespace {
PDerivativeWorkspace cpu_derivatives_from_local(
    const WorkerInput& w, const Matrix& knots, const GlobalParameters& p,
    const LocalComputationResult& local) {
  const auto kd=cpu_matern(knots,knots,p.nu2,p.alpha,p.range);
  const auto cd=cpu_matern(w.sites,knots,p.nu2,p.alpha,p.range);
  const auto sd=cpu_matern(w.sites,w.sites,p.nu2,p.alpha,p.range);
  const auto kl=cholesky_lower(kd.value);
  auto rs=[&](const Matrix& x){return transpose(solve_cholesky(kl,transpose(x)));};
  Matrix ba=rs(cd.alpha);axpy(ba,rs(matmul(local.B,kd.alpha)),-1);
  Matrix br=rs(cd.range);axpy(br,rs(matmul(local.B,kd.range)),-1);
  auto b2=[&](const Matrix& cij,const Matrix& ci,const Matrix& kj,const Matrix& bj,const Matrix& ki,const Matrix& kij){
    Matrix out=rs(cij);axpy(out,rs(matmul(rs(ci),kj)),-1);
    axpy(out,rs(matmul(bj,ki)),-1);axpy(out,rs(matmul(local.B,kij)),-1);
    axpy(out,rs(matmul(rs(matmul(local.B,ki)),kj)));return out;
  };
  PDerivativeWorkspace d;
  d.b_second={kd.aa,kd.ar,kd.rr,cd.aa,cd.ar,cd.rr,
    b2(cd.aa,cd.alpha,kd.alpha,ba,kd.alpha,kd.aa),
    b2(cd.ar,cd.alpha,kd.range,br,kd.alpha,kd.ar),
    b2(cd.ra,cd.range,kd.alpha,ba,kd.range,kd.ra),
    b2(cd.rr,cd.range,kd.range,br,kd.range,kd.rr)};
  auto a1=[&](Matrix out,const Matrix& bi,const Matrix& ki){
    subtract_triples(out,{{&bi,&kd.value,&local.B},{&local.B,&ki,&local.B},
                          {&local.B,&kd.value,&bi}});return out;};
  auto a2=[&](Matrix out,const Matrix& bi,const Matrix& bj,const Matrix& bij,const Matrix& ki,const Matrix& kj,const Matrix& kij){
    subtract_triples(out,{{&bij,&kd.value,&local.B},{&bi,&kj,&local.B},
      {&bi,&kd.value,&bj},{&bj,&ki,&local.B},{&local.B,&kij,&local.B},
      {&local.B,&ki,&bj},{&bj,&kd.value,&bi},{&local.B,&kj,&bi},
      {&local.B,&kd.value,&bij}});return out;};
  const auto& b=d.b_second;
  d.a_second={a2(sd.aa,ba,ba,b.baa,kd.alpha,kd.alpha,kd.aa),
    a2(sd.ar,ba,br,b.bar,kd.alpha,kd.range,kd.ar),
    a2(sd.ra,br,ba,b.bra,kd.range,kd.alpha,kd.ra),
    a2(sd.rr,br,br,b.brr,kd.range,kd.range,kd.rr)};
  Matrix av=local.M;add_diagonal(av,-1);av=scaled(av,1./p.delta);
  d.m_first={av,scaled(a1(sd.alpha,ba,kd.alpha),p.delta),scaled(a1(sd.range,br,kd.range),p.delta)};
  auto da=scaled(d.m_first.alpha,1./p.delta),dr=scaled(d.m_first.range,1./p.delta);
  d.m_second={Matrix(w.sites.rows,w.sites.rows),da,dr,da,
    scaled(d.a_second.aa,p.delta),scaled(d.a_second.ar,p.delta),dr,
    scaled(d.a_second.ra,p.delta),scaled(d.a_second.rr,p.delta)};
  return d;
}
} // namespace

PDerivativeWorkspace cpu_derivatives(const WorkerInput& w,const Matrix& knots,const GlobalParameters& p) {
  const auto local=cpu_local(w,knots,p);
  return cpu_derivatives_from_local(w,knots,p,local);
}

void cpu_update_mu_sigma(GlobalParameters& p,const Matrix& knots,const MuSigmaLocalSummary& mean,int workers) {
  if(workers<=0||mean.mu.rows!=knots.rows||mean.mu.cols!=1||mean.sigma.rows!=knots.rows||mean.sigma.cols!=knots.rows)
    throw std::runtime_error("CPU mu/Sigma update dimensions");
  auto precision=inverse_spd(covariance_value(knots,knots,p.nu2,p.alpha,p.range));
  axpy(precision,mean.sigma,workers*p.delta);
  p.sigma=inverse_spd(precision);p.mu=scaled(matmul(p.sigma,mean.mu),-workers*p.delta);
}
PSynchronousRoundResult cpu_newton(const GlobalParameters& p,const Matrix& gradient,const Matrix& hessian) {
  if(gradient.rows!=3||gradient.cols!=1||hessian.rows!=3||hessian.cols!=3)throw std::runtime_error("CPU Newton dimensions");
  for(double v:gradient.a)if(!std::isfinite(v))throw std::runtime_error("non-finite Newton gradient");
  for(double v:hessian.a)if(!std::isfinite(v))throw std::runtime_error("non-finite Newton Hessian");
  auto eigen=symmetric_eigen3(hessian);Matrix modified(3,3);
  for(int k=0;k<3;++k){double v=std::max(std::abs(eigen.values[k]),.01);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)modified(i,j)+=eigen.vectors(i,k)*v*eigen.vectors(j,k);}
  auto direction=solve_spd(modified,gradient);
  for(double v:direction.a)if(!std::isfinite(v))throw std::runtime_error("non-finite Newton direction");
  auto state=p;double step=1.;int halves=0;
  for(;;){state.delta=p.delta-step*direction(0,0);state.alpha=p.alpha-step*direction(1,0);state.range=p.range-step*direction(2,0);
    if(state.delta>0&&state.alpha>0&&state.range>0)break;
    if(++halves>1074)throw std::runtime_error("Newton positivity halving exhausted");step*=.5;}
  return {state,step,halves};
}
} // namespace fedspatial
