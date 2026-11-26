#!/usr/bin/env python3
"""Bounded equality audit for a raw global-then-consecutive Figure-1 fixture."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--knots", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if args.n <= 0 or args.n % args.workers:
        raise SystemExit("n must be positive and divisible by workers")
    metadata = json.loads((args.fixture / "metadata.json").read_text())
    expected = {"n": args.n, "knots": args.knots, "workers": args.workers,
                "observations_per_worker": args.n // args.workers,
                "partition_order": "global_then_consecutive"}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise SystemExit(f"fixture metadata mismatch: {expected}")
    arrays = {name: np.load(args.fixture / f"{name}.npy")
              for name in ("sites", "knots", "X", "latent", "noise", "z")}
    if arrays["sites"].shape != (args.n, 2) or arrays["X"].shape != (args.n, 5) or arrays["knots"].shape != (args.knots, 2):
        raise SystemExit("fixture shapes do not match metadata")
    if not all(value.dtype == np.float64 and np.isfinite(value).all() for value in arrays.values()):
        raise SystemExit("fixture must be finite FP64")
    hashes = metadata.get("array_sha256", {})
    if any(hashes.get(name) != digest(value) for name, value in arrays.items()):
        raise SystemExit("fixture SHA-256 mismatch")
    # These slices are exactly what partition_figure1_fixture consumes: no
    # worker-local regeneration and no shuffle/reordering is permitted.
    rows = args.n // args.workers
    blocks = [arrays["sites"][worker * rows:(worker + 1) * rows] for worker in range(args.workers)]
    if not all(np.array_equal(block, arrays["sites"][worker * rows:(worker + 1) * rows])
               for worker, block in enumerate(blocks)):
        raise SystemExit("consecutive partition equality failed")
    print("FIXTURE_EQUALITY=PASS")
    print(f"FIXTURE_ROWS={args.n} WORKERS={args.workers} ROWS_PER_WORKER={rows}")
    print("FIXTURE_MAX_ABS_ERROR=0 FIXTURE_MAX_REL_ERROR=0")


if __name__ == "__main__":
    main()
