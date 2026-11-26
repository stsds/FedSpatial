#include "fedspatial/sync_cpu.hpp"

#if defined(FEDSPATIAL_CRAY_LIBSCI)
#include <lapacke.h>
#elif defined(FEDSPATIAL_MKL_LOCAL)
#include <mkl_lapacke.h>
#else
#error "The ExaGeoStat fixture exporter requires the configured LAPACKE provider"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using fedspatial::Matrix;
using fedspatial::write_matrix;

namespace {
std::uint64_t spread_bits(std::uint64_t x) {
  x &= 0x000000000000ffffULL;
  x = (x ^ (x << 24)) & 0x000000ff000000ffULL;
  x = (x ^ (x << 12)) & 0x000f000f000f000fULL;
  x = (x ^ (x << 6)) & 0x0303030303030303ULL;
  x = (x ^ (x << 3)) & 0x1111111111111111ULL;
  return x;
}

std::uint64_t reverse_spread_bits(std::uint64_t x) {
  x &= 0x1111111111111111ULL;
  x = (x ^ (x >> 3)) & 0x0303030303030303ULL;
  x = (x ^ (x >> 6)) & 0x000f000f000f000fULL;
  x = (x ^ (x >> 12)) & 0x000000ff000000ffULL;
  x = (x ^ (x >> 24)) & 0x000000000000ffffULL;
  return x;
}

int positive(const std::string& value, const char* name) {
  std::size_t used=0;
  const long long result=std::stoll(value,&used);
  if(used!=value.size()||result<=0||result>std::numeric_limits<int>::max())
    throw std::runtime_error(std::string("invalid ")+name);
  return static_cast<int>(result);
}

struct Options { int n=0,workers=0,m=400,seed=1; fs::path output; };
Options parse(int argc,char** argv) {
  Options o;
  for(int i=1;i<argc;++i) {
    std::string arg=argv[i],value;
    const auto equals=arg.find('=');
    if(equals!=std::string::npos){value=arg.substr(equals+1);arg.resize(equals);}
    else {if(++i==argc)throw std::runtime_error("missing value for "+arg);value=argv[i];}
    if(arg=="--n")o.n=positive(value,"n");
    else if(arg=="--workers")o.workers=positive(value,"workers");
    else if(arg=="--m")o.m=positive(value,"m");
    else if(arg=="--seed")o.seed=positive(value,"seed");
    else if(arg=="--output-dir")o.output=value;
    else throw std::runtime_error("unknown option "+arg);
  }
  if(o.n<=0||o.workers<=0||o.output.empty())throw std::runtime_error("require --n, --workers and --output-dir");
  if(o.n%o.workers)throw std::runtime_error("n must be divisible by workers");
  return o;
}
}

int main(int argc,char** argv) try {
  const auto o=parse(argc,argv);
  std::srand(o.seed);
  const int side=static_cast<int>(std::ceil(std::sqrt(static_cast<double>(o.n))));
  std::vector<std::uint64_t> morton;
  morton.reserve(o.n);
  for(int i=0;i<side&&static_cast<int>(morton.size())<o.n;++i) {
    for(int j=0;j<side&&static_cast<int>(morton.size())<o.n;++j) {
      const auto uniform=[] {
        const double u=static_cast<double>(std::rand())/(1.0+RAND_MAX);
        return -0.4+0.8*u;
      };
      const double x=(i+0.5+uniform())/side;
      const double y=(j+0.5+uniform())/side;
      const auto xi=static_cast<std::uint16_t>(x*UINT16_MAX+0.5);
      const auto yi=static_cast<std::uint16_t>(y*UINT16_MAX+0.5);
      morton.push_back((spread_bits(yi)<<1)+spread_bits(xi));
    }
  }
  std::sort(morton.begin(),morton.end());

  // Match ParsecGenerator exactly.  It asks LAPACKE for 100*N values and
  // initializes descriptor Z from the second N-value segment.
  if(static_cast<std::size_t>(o.n)>std::numeric_limits<std::size_t>::max()/100)
    throw std::runtime_error("random workspace overflow");
  std::vector<double> random_values(static_cast<std::size_t>(o.n)*100);
  lapack_int random_seed[4]={o.seed,o.seed,o.seed,1};
  const lapack_int info=LAPACKE_dlarnv(3,random_seed,static_cast<lapack_int>(random_values.size()),random_values.data());
  if(info!=0)throw std::runtime_error("LAPACKE_dlarnv failed: "+std::to_string(info));

  fs::create_directories(o.output);
  const int local_n=o.n/o.workers;
  for(int worker=0;worker<o.workers;++worker) {
    Matrix data(local_n,4);
    for(int local=0;local<local_n;++local) {
      const int global=worker*local_n+local;
      data(local,0)=static_cast<double>(reverse_spread_bits(morton[global]))/UINT16_MAX;
      data(local,1)=static_cast<double>(reverse_spread_bits(morton[global]>>1))/UINT16_MAX;
      data(local,2)=random_values[static_cast<std::size_t>(o.n)+global];
      data(local,3)=0.0; // ignored by --zero-mean; retained for the Matrix fixture format
    }
    if(!write_matrix((o.output/("worker_"+std::to_string(worker)+".txt")).string(),data))
      throw std::runtime_error("worker fixture write failed");
  }

  const int knot_rows=std::max(1,static_cast<int>(std::floor(std::sqrt(o.m))));
  const int knot_cols=(o.m+knot_rows-1)/knot_rows;
  Matrix knots(o.m,2);
  int k=0;
  for(int i=0;i<knot_rows&&k<o.m;++i)for(int j=0;j<knot_cols&&k<o.m;++j,++k) {
    knots(k,0)=(i+0.5)/knot_rows;
    knots(k,1)=(j+0.5)/knot_cols;
  }
  if(!write_matrix((o.output/"knots.txt").string(),knots))throw std::runtime_error("knot fixture write failed");
  std::ofstream metadata(o.output/"generator_manifest.txt");
  metadata<<"generator=ExaGeoStat_ParsecGenerator_seed1_compatible\n"
          <<"n="<<o.n<<"\nworkers="<<o.workers<<"\nlocal_n="<<local_n
          <<"\nm="<<o.m<<"\nseed="<<o.seed
          <<"\nlocation_order=morton_uint16\n"
          <<"z_source=LAPACKE_dlarnv_idist3_segment_N_to_2N_of_100N\n"
          <<"mean_model=fixed_zero\n";
  if(!metadata)throw std::runtime_error("manifest write failed");
  std::cout<<"EXAGEOSTAT_SEED1_FIXTURE_EXPORT=PASS\n";
  return 0;
} catch(const std::exception& e) {
  std::cerr<<e.what()<<'\n';return 1;
}
