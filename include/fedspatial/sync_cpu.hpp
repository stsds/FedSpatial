#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>

namespace fedspatial {
struct Matrix {
  int rows{}, cols{};
  std::vector<double> a;
  // Non-negative only for an opaque matrix whose rows remain owned by the
  // distributed CPU service ranks. Ordinary and GPU matrices keep -1.
  int distributed_handle{-1};
  Matrix() = default;
  Matrix(int r, int c, double v = 0.0) : rows(r), cols(c) {
    if (r < 0 || c < 0 || (c && size_t(r) > a.max_size()/size_t(c)))
      throw std::runtime_error("invalid matrix dimensions");
    a.assign(size_t(r)*size_t(c), v);
  }
  double& operator()(int r, int c) { return a[size_t(r)*size_t(cols)+size_t(c)]; }
  double operator()(int r, int c) const { return a[size_t(r)*size_t(cols)+size_t(c)]; }
};
Matrix matern15(const Matrix& x, const Matrix& y, double alpha, double range);
Matrix matmul(const Matrix& a, const Matrix& b);
// Evaluate (a*b)*c without materializing the first product on the service
// leader when the distributed CPU backend is active.
Matrix matmul3(const Matrix& a, const Matrix& b, const Matrix& c);
Matrix transpose(const Matrix& a);
Matrix inverse_spd(const Matrix& a);
Matrix solve_spd(const Matrix& a, const Matrix& rhs);
Matrix cholesky_lower(const Matrix& a);
Matrix solve_cholesky(const Matrix& lower, const Matrix& rhs);
Matrix subtract(const Matrix& a, const Matrix& b);
Matrix identity(int n);
void add_diagonal(Matrix& a, double v);
double max_abs_diff(const Matrix& a, const Matrix& b);
struct LocalP { Matrix K, B, M, Minv, btMinvb, btMinve; };
LocalP make_local_p(const Matrix& sites, const Matrix& knots, const Matrix& z, const Matrix& x,
                    const Matrix& gamma, double delta, double alpha, double range);
bool read_matrix(const std::string& path, Matrix* out);
bool write_matrix(const std::string& path, const Matrix& value);
}
