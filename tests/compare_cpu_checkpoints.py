#!/usr/bin/env python3
"""Compare complete staged FP64 state and Newton metadata, without GPU execution."""
import argparse
import csv
from pathlib import Path
import math

p = argparse.ArgumentParser()
p.add_argument('actual', type=Path)
p.add_argument('reference', type=Path)
p.add_argument('--tolerance', type=float, default=1e-10)
a = p.parse_args()
with a.actual.open() as f:
    reader = csv.reader(f)
    header = next(reader)
    actual = list(reader)
with a.reference.open() as f:
    reader = csv.reader(f)
    next(reader)
    reference = list(reader)
if len(actual) != len(reference):
    raise SystemExit('checkpoint count mismatch')
errors = {}
for x, y in zip(actual, reference):
    if x[:2] != y[:2] or len(x) != len(y):
        raise SystemExit('checkpoint stage/order/shape mismatch')
    for i, (u, v) in enumerate(zip(x[2:], y[2:])):
        u, v = float(u), float(v)
        if not math.isfinite(u) or not math.isfinite(v):
            raise SystemExit('non-finite checkpoint')
        diff = abs(u-v)
        column = header[i+2]
        key = ('gamma' if column.startswith('gamma') else
               'mu' if column.startswith('mu') else
               'Sigma' if column.startswith('sigma') else
               'step' if column == 'accepted_step' else
               'halves' if column == 'halving_count' else column)
        old = errors.get(key, (0, 0))
        errors[key] = max(old[0], diff), max(old[1], diff/max(1, abs(v)))
        if diff > (0 if key in ('step', 'halves') else a.tolerance):
            raise SystemExit(f'{x[:2]} {key}: {u} != {v} (error {diff})')
for name, (absolute, relative) in errors.items():
    print(f'{name} MAX_ABS={absolute:.17g} MAX_REL={relative:.17g}')
print('CPU_CHECKPOINT_PARITY=PASS')
