# FedSpatial

**Scaling Spatial Inference Beyond Centralized Computing on Distributed Systems**

FedSpatial is a research implementation of synchronous decentralized inference
for large spatial Gaussian-process models. It partitions observations across
logical workers, keeps raw observations local, and exchanges compact model
summaries during each synchronous round. The repository contains matching CPU
and NVIDIA GPU backends, MPI orchestration, deterministic reference fixtures,
and numerical parity tests.

## Highlights

- **CPU and GPU implementations:** C++17/OpenMP for CPUs and CUDA with
  cuBLAS/cuSOLVER for NVIDIA GPUs.
- **Distributed execution:** MPI workers with deterministic, worker-ordered
  aggregation and explicit communication accounting.
- **Hierarchical CPU workers:** optional ScaLAPACK support through oneMKL or
  Cray LibSci for multi-rank CPU workers.
- **Reproducible numerics:** stored FP64 fixtures cover covariance kernels,
  local summaries, gradients, Hessians, Newton updates, and full synchronous
  trajectories.
- **Matérn models:** the production driver supports smoothness values
  \(\nu\in\{0.5,1.5,2.5\}\).

## Execution model

```text
worker 0 data ─┐
worker 1 data ─┼─ local CPU/GPU summaries ── deterministic aggregation
      ...      │                                      │
worker J data ─┘                       updated global parameters
                                                       │
                                      broadcast to every worker
```

Each round updates the regression and low-rank process state in three stages:
`mu/sigma`, `gamma`, and the joint covariance-parameter Newton step. Raw data
remain at their worker; only summary statistics and the updated model state
cross worker boundaries.

## Repository layout

| Path | Purpose |
| --- | --- |
| `include/fedspatial/` | Public data structures and CPU/GPU interfaces |
| `src/` | CPU, CUDA, collective, and distributed kernels |
| `apps/` | Production MPI drivers, parity programs, and profiling tools |
| `tests/` | Deterministic validation scripts and stored FP64 fixtures |
| `scripts/` | Experiment aggregation and plotting utilities |

## Requirements

### Common

- CMake 3.22 or newer
- A C++17 compiler
- OpenMP

### Distributed CPU backend

- MPI with C++ bindings (OpenMPI and Cray MPICH are supported configurations)
- Optional: ScaLAPACK through oneMKL or Cray LibSci

### GPU backend

- NVIDIA CUDA Toolkit
- cuBLAS and cuSOLVER
- MPI for multi-GPU/multi-node execution

Python 3 is used only by validation and plotting utilities. The core library
and executables do not require Python.

## Build and test

### Portable CPU build

This configuration has no MPI or CUDA dependency and runs the local parity
tests:

```bash
cmake -S . -B build/cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DFEDSPATIAL_ENABLE_MPI=OFF \
  -DFEDSPATIAL_ENABLE_CUDA=OFF
cmake --build build/cpu -j
ctest --test-dir build/cpu --output-on-failure
```

### MPI CPU build

```bash
cmake -S . -B build/cpu-mpi \
  -DCMAKE_BUILD_TYPE=Release \
  -DFEDSPATIAL_ENABLE_MPI=ON \
  -DFEDSPATIAL_ENABLE_CUDA=OFF
cmake --build build/cpu-mpi -j

mpirun -np 2 build/cpu-mpi/sync_mpi_trajectory \
  --backend=cpu \
  --workers=2 \
  --ranks-per-worker=1 \
  --cpus-per-rank=1 \
  --rounds=2 \
  --small-fixture \
  --fixture-dir=tests/fixtures/sync_gpu \
  --output-dir=results/cpu-smoke
```

For distributed dense linear algebra, add
`-DFEDSPATIAL_ENABLE_SCALAPACK=ON` and select either
`-DFEDSPATIAL_SCALAPACK_PROVIDER=MKL` or
`-DFEDSPATIAL_SCALAPACK_PROVIDER=CRAY_LIBSCI`.

### CUDA build

Set the CUDA architecture to match the target GPU (for example, `80` for
A100 or `90` for H100/GH200):

```bash
cmake -S . -B build/gpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DFEDSPATIAL_ENABLE_MPI=ON \
  -DFEDSPATIAL_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build/gpu -j
ctest --test-dir build/gpu --output-on-failure
```

The unified trajectory driver accepts `--backend=gpu` in CUDA-enabled builds.
Use one MPI rank per GPU and one GPU per logical worker:

```bash
mpirun -np 2 build/gpu/sync_mpi_trajectory \
  --backend=gpu \
  --workers=2 \
  --ranks-per-worker=1 \
  --cpus-per-rank=1 \
  --rounds=2 \
  --small-fixture \
  --fixture-dir=tests/fixtures/sync_gpu \
  --output-dir=results/gpu-smoke
```

Run `build/gpu/sync_mpi_trajectory --help` for the complete production-driver
interface. The standalone programs in `apps/` isolate individual numerical
blocks and are useful when porting to a new GPU architecture.

## Input and output

The production fixture loader accepts NumPy FP64 arrays named `sites.npy`,
`z.npy`, `X.npy`, and `knots.npy`; optional `latent.npy` and `noise.npy` arrays
are used by reference workflows. Text fixtures use `worker_<id>.txt` and
`knots.txt`, with matrix dimensions in the first line followed by row-major
values. See `tests/fixtures/` for small, version-controlled examples.

Every trajectory run writes:

- `checkpoints.csv` — the global state after each stage and round;
- `timing.csv` — local computation, aggregation, update, and broadcast times;
- `peak_rss.csv` — per-rank peak resident memory; and
- `manifest.txt` — topology, backend, precision, input, and communication data.

Large scientific datasets, generated results, profiler traces, and cluster
credentials are intentionally excluded from this repository.

## Reproducibility

The committed fixtures are small deterministic validation cases, not research
datasets. A recommended validation sequence for a new system is:

1. build and run the CPU CTest suite;
2. build the CUDA backend and run `sync_gpu_parity`;
3. run the two-worker CPU and GPU smoke commands above; and
4. compare complete checkpoint trajectories before launching large jobs.

Generated results should be written below `results/`, which is ignored by Git.

## Current limitations

- The CUDA production path currently uses one GPU per logical worker; local
  covariance factorization across multiple GPUs in a single worker is not yet
  implemented.
- The optional distributed CPU backend parallelizes dominant dense operations,
  but some derivative workspace remains leader-owned and quadratic in the
  worker's local observation count.
- This is research software. Validate numerical trajectories and resource
  requirements on a small case before a production run.

## Team and citation

FedSpatial is developed at King Abdullah University of Science and Technology
(KAUST) by Sameh Abdulah, David Keyes, Jianwei, and collaborators.

If you use this software, please cite the accompanying paper,
*Scaling Spatial Inference Beyond Centralized Computing on Distributed
Systems*. A final BibTeX record will be added when the archival version is
available.

## License

No open-source license has been attached yet. Until a license is added, the
source is publicly viewable but standard copyright restrictions apply.
