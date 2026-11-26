#include "fedspatial/figure1_fixture.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>

namespace fedspatial {
namespace {
Matrix read_npy_f64_c(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open fixture " + path.string());
  char magic[6]{}; in.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY") throw std::runtime_error("not an NPY file: " + path.string());
  unsigned char version[2]{}; in.read(reinterpret_cast<char*>(version), 2);
  std::size_t header_size = 0;
  if (version[0] == 1) { std::uint16_t n{}; in.read(reinterpret_cast<char*>(&n), sizeof(n)); header_size = n; }
  else if (version[0] == 2 || version[0] == 3) { std::uint32_t n{}; in.read(reinterpret_cast<char*>(&n), sizeof(n)); header_size = n; }
  else throw std::runtime_error("unsupported NPY version: " + path.string());
  std::string header(header_size, '\0'); in.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (!in || header.find("'descr': '<f8'") == std::string::npos || header.find("'fortran_order': False") == std::string::npos)
    throw std::runtime_error("fixture must be C-order little-endian FP64: " + path.string());
  std::smatch match;
  const std::regex two_dim("'shape':[[:space:]]*\\(([[:digit:]]+),[[:space:]]*([[:digit:]]+),?\\)");
  const std::regex one_dim("'shape':[[:space:]]*\\(([[:digit:]]+),?\\)");
  int rows = 0, cols = 0;
  if (std::regex_search(header, match, two_dim)) { rows = std::stoi(match[1]); cols = std::stoi(match[2]); }
  else if (std::regex_search(header, match, one_dim)) { rows = std::stoi(match[1]); cols = 1; }
  else throw std::runtime_error("unsupported NPY shape: " + path.string());
  Matrix result(rows, cols);
  in.read(reinterpret_cast<char*>(result.a.data()), static_cast<std::streamsize>(result.a.size() * sizeof(double)));
  if (!in) throw std::runtime_error("truncated NPY data: " + path.string());
  for (double v : result.a) if (!std::isfinite(v)) throw std::runtime_error("non-finite fixture value: " + path.string());
  return result;
}
}  // namespace

bool load_figure1_python_fixture(const std::filesystem::path& directory, Figure1Fixture* fixture) {
  if (!fixture) return false;
  fixture->sites = read_npy_f64_c(directory / "sites.npy");
  fixture->knots = read_npy_f64_c(directory / "knots.npy");
  fixture->x = read_npy_f64_c(directory / "X.npy");
  fixture->latent = read_npy_f64_c(directory / "latent.npy");
  fixture->noise = read_npy_f64_c(directory / "noise.npy");
  fixture->z = read_npy_f64_c(directory / "z.npy");
  const int n = fixture->sites.rows;
  if (n <= 0 || fixture->sites.cols != 2 || fixture->knots.rows <= 0 ||
      fixture->knots.cols != 2 || fixture->x.rows != n || fixture->x.cols <= 0 ||
      fixture->latent.rows != n || fixture->latent.cols != 1 ||
      fixture->noise.rows != n || fixture->noise.cols != 1 || fixture->z.rows != n ||
      fixture->z.cols != 1)
    throw std::runtime_error("inconsistent FP64 spatial fixture dimensions in " + directory.string());
  return true;
}

std::vector<WorkerInput> partition_figure1_fixture(const Figure1Fixture& fixture, int workers) {
  if (workers <= 0 || fixture.sites.rows % workers != 0) throw std::runtime_error("fixture cannot be partitioned evenly");
  const int local_n = fixture.sites.rows / workers;
  std::vector<WorkerInput> result; result.reserve(workers);
  for (int worker = 0; worker < workers; ++worker) {
    WorkerInput local{Matrix(local_n, 2), Matrix(local_n, 1), Matrix(local_n, fixture.x.cols)};
    for (int row = 0; row < local_n; ++row) {
      const int source = worker * local_n + row;
      for (int col = 0; col < 2; ++col) local.sites(row, col) = fixture.sites(source, col);
      local.z(row, 0) = fixture.z(source, 0);
      for (int col = 0; col < fixture.x.cols; ++col) local.x(row, col) = fixture.x(source, col);
    }
    result.push_back(std::move(local));
  }
  return result;
}
}  // namespace fedspatial
