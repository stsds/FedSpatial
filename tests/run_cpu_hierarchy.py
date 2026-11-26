#!/usr/bin/env python3
"""Exercise the real executor, reference checkpoints, and topology invariance."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--build-dir', type=Path, required=True)
p.add_argument('--source-dir', type=Path, default=Path(__file__).resolve().parents[1])
p.add_argument('--mpiexec', default='mpirun')
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
source, build, out = a.source_dir.resolve(), a.build_dir.resolve(), a.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
fixtures = source/'tests/fixtures/sync_gpu'
env = os.environ.copy()
env['OMPI_MCA_rmaps_base_oversubscribe'] = '1'
env['OMPI_MCA_hwloc_base_binding_policy'] = 'none'
# Shared-memory transport avoids unnecessary UCX registration on the CPU node.
env['OMPI_MCA_pml'] = 'ob1'
env['OMPI_MCA_btl'] = 'self,vader'
# The small hierarchy gate intentionally oversubscribes to test collectives;
# it is a correctness test and makes no throughput claims.
cases = [(1,1,1), (2,1,1), (3,1,1), (10,1,1), (2,2,1),
         (2,2,2), (2,4,2), (3,2,2), (10,2,1)]
baselines = {}
for workers, ranks, threads in cases:
    dest = out/f'{workers}x{ranks}x{threads}'
    cmd = [a.mpiexec, '-np', str(workers*ranks), str(build/'sync_mpi_trajectory'),
           '--backend=cpu', f'--ranks-per-worker={ranks}', f'--cpus-per-rank={threads}',
           f'--workers={workers}', '--rounds=2', '--small-fixture',
           f'--fixture-dir={fixtures}', f'--output-dir={dest}']
    print('COMMAND', ' '.join(cmd), flush=True)
    result = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=90)
    (out/f'{workers}x{ranks}x{threads}.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise SystemExit(result.stdout+result.stderr)
    subprocess.run([sys.executable, str(source/'tests/compare_cpu_checkpoints.py'),
                    str(dest/'checkpoints.csv'), str(fixtures/f'N_WORKER_{workers}_P_CHECKPOINTS.csv')],
                   check=True, timeout=30)
    raw = (dest/'checkpoints.csv').read_bytes()
    if workers in baselines and baselines[workers] != raw:
        if 'workspace=leader_dense' not in (dest/'manifest.txt').read_text():
            raise SystemExit(f'topology changed FP64 state: {workers}x{ranks}x{threads}')
        # ScaLAPACK changes FP64 summation order; retain complete-state and
        # exact Newton-decision gates instead of imposing bitwise scalar parity.
        baseline_path = out/f'{workers}x1x1/checkpoints.csv'
        subprocess.run([sys.executable, str(source/'tests/compare_cpu_checkpoints.py'),
                        str(dest/'checkpoints.csv'), str(baseline_path)], check=True)

    baselines.setdefault(workers, raw)
    if (workers,ranks,threads) == (2,2,2):
        repeat_baseline = raw
    print(f'CASE {workers}x{ranks}x{threads} PASS', flush=True)
# Same process topology, fresh MPI processes: exact byte repeatability.
dest = out/'2x2x2-repeat'
subprocess.run([a.mpiexec, '-np', '4', str(build/'sync_mpi_trajectory'), '--backend=cpu',
                '--ranks-per-worker=2', '--cpus-per-rank=2', '--workers=2', '--rounds=2',
                '--small-fixture', f'--fixture-dir={fixtures}', f'--output-dir={dest}'],
               env=env, check=True, timeout=90)
if (dest/'checkpoints.csv').read_bytes() != repeat_baseline:
    raise SystemExit('fresh-process determinism failed')
print('CPU_HIERARCHY_CASES=9 CPU_REPEAT=BYTE_IDENTICAL CPU_TOPOLOGY_INVARIANCE=FULL_STATE_TOLERANCE_AND_EXACT_NEWTON')
