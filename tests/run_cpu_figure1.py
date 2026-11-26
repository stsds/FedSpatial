#!/usr/bin/env python3
"""Validate text and production NPY input paths against stored FP64 trajectory."""
import argparse
import csv
import os
from pathlib import Path
import struct
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--build-dir', type=Path, required=True)
p.add_argument('--source-dir', type=Path, default=Path(__file__).resolve().parents[1])
p.add_argument('--mpiexec', default='mpirun')
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
source, out = a.source_dir.resolve(), a.output_dir.resolve()
fixture = source/'tests/fixtures/figure1_reference'
raw = out/'npy-fixture'
raw.mkdir(parents=True, exist_ok=True)
def matrix(path):
    values = path.read_text().split()
    rows, cols = map(int, values[:2])
    data = list(map(float, values[2:]))
    assert len(data) == rows*cols
    return rows, cols, data

def npy(name, rows, cols, data):
    # Exact little-endian FP64 serialization; no generation, RNG or NumPy needed.
    header = repr({'descr': '<f8', 'fortran_order': False, 'shape': (rows, cols)})
    header = (header + ' ' * ((64-(10+len(header)+1)%64)%64) + '\n').encode('ascii')
    with (raw/f'{name}.npy').open('wb') as f:
        f.write(b'\x93NUMPY\x01\x00'+struct.pack('<H', len(header))+header)
        f.write(struct.pack(f'<{len(data)}d', *data))
rows, cols, data = matrix(fixture/'global_data.txt')
kr, kc, knots = matrix(fixture/'knots.txt')
npy('knots', kr, kc, knots)
npy('sites', rows, 2, [data[i*cols+j] for i in range(rows) for j in range(2)])
npy('X', rows, cols-3, [data[i*cols+j] for i in range(rows) for j in range(3,cols)])
npy('z', rows, 1, [data[i*cols+2] for i in range(rows)])
npy('latent', rows, 1, [0.]*rows)
npy('noise', rows, 1, [0.]*rows)
env = os.environ.copy()
env.update(OMPI_MCA_rmaps_base_oversubscribe='1', OMPI_MCA_hwloc_base_binding_policy='none',
           OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader')
for mode, directory, extra in [('text',fixture,['--text-fixture']),('npy',raw,[])]:
    cmd = [a.mpiexec, '-np', '10', str(a.build_dir.resolve()/'sync_mpi_trajectory'),
           '--backend=cpu', '--ranks-per-worker=1', '--cpus-per-rank=1', '--workers=10',
           '--rounds=2', f'--fixture-dir={directory}', f'--output-dir={out/mode}', *extra]
    print('COMMAND', ' '.join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True, timeout=90)
if (out/'text/checkpoints.csv').read_bytes() != (out/'npy/checkpoints.csv').read_bytes():
    raise SystemExit('text and NPY input paths produced different state')
with (out/'npy/checkpoints.csv').open() as f:
    actual = list(csv.DictReader(f))
with (fixture/'low_rank_trajectory.csv').open() as f:
    reference = list(csv.DictReader(f))
errors = {}
for r in actual:
    if r['stage'] != 'delta_theta':
        continue
    ref = reference[int(r['round'])+1]
    for key, refkey in [(f'gamma{i}',f'gamma_{i}') for i in range(5)]+[('delta','delta'),('alpha','alpha'),('range','beta')]:
        value, target = float(r[key]), float(ref[refkey])
        diff = abs(value-target)
        old = errors.get(key, (0,0))
        errors[key] = max(old[0], diff), max(old[1], diff/max(1, abs(target)))
        if diff > 1e-10:
            raise SystemExit(f'{key}: {value} != {target}')
for key, (absolute, relative) in errors.items():
    print(f'{key} MAX_ABS={absolute:.17g} MAX_REL={relative:.17g}')
print('FIGURE1_CPU_TRAJECTORY=PASS CPU_INPUT_PATHS=BYTE_IDENTICAL')
