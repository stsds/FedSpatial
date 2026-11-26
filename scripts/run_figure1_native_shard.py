#!/usr/bin/env python3
"""Fail-fast, resumable Figure-1 shard: raw Python fixture -> native CUDA case."""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def valid_native_estimates(path):
    try:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        return len(rows) == 2 and {row.get("method") for row in rows} == {"P", "IND"} and all(row.get("status") == "OK" for row in rows)
    except (OSError, csv.Error):
        return False


def run_logged(parts, log, env=None):
    with log.open("a") as stream:
        stream.write("+ " + " ".join(map(str, parts)) + "\n")
        stream.flush()
        subprocess.run(list(map(str, parts)), check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenario-start", type=int, required=True); parser.add_argument("--scenario-end", type=int, required=True)
    parser.add_argument("--rep-start", type=int, required=True); parser.add_argument("--rep-end", type=int, required=True)
    parser.add_argument("--worker-devices", default="0,0,0,0,0,0,0,0,0,0")
    parser.add_argument("--omp-threads", type=int, default=0,
                        help="native OpenMP threads per process; 0 preserves environment")
    parser.add_argument("--case-modulus", type=int, default=1); parser.add_argument("--case-remainder", type=int, default=0)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--no-aggregate", action="store_true"); parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.case_modulus < 1 or not 0 <= args.case_remainder < args.case_modulus:
        raise SystemExit("invalid --case-modulus/--case-remainder")
    if args.omp_threads < 0: raise SystemExit("invalid --omp-threads")
    root = Path(__file__).resolve().parents[2]
    exporter = root / "sync_gpu/tests/export_figure1_python_fixture.py"
    executable = args.build_dir / "sync_gpu_figure1_fresh_case"
    if not executable.is_file(): raise SystemExit(f"missing native executable: {executable}")
    cases, raw, logs = args.output_root / "cases", args.output_root / "raw", args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    launcher_log = logs / f"shard_{args.case_remainder}_of_{args.case_modulus}.log"
    native_env = None
    if args.omp_threads:
        native_env = os.environ.copy()
        native_env["OMP_NUM_THREADS"] = str(args.omp_threads)
        native_env["OMP_DYNAMIC"] = "FALSE"
    launched = skipped = 0
    for scenario in range(args.scenario_start, args.scenario_end + 1):
        for replication in range(args.rep_start, args.rep_end + 1):
            if (scenario * 100 + replication) % args.case_modulus != args.case_remainder: continue
            case = cases / f"scenario_{scenario:02d}" / f"rep_{replication:03d}"
            native = case / "native"; complete = native / "COMPLETE.json"
            estimates = native / "native_estimates.csv"
            if args.resume and complete.is_file() and valid_native_estimates(estimates):
                skipped += 1; continue
            case.mkdir(parents=True, exist_ok=True)
            lock = case / ".native_case_lock"
            try:
                os.mkdir(lock)
            except FileExistsError:
                raise RuntimeError(f"case is already owned by another launcher: {scenario}/{replication}")
            case_log = logs / f"scenario_{scenario:02d}_rep_{replication:03d}.log"
            temporary = case / f"native.tmp.{os.getpid()}"
            try:
                run_logged((sys.executable, exporter, "--scenario-id", scenario, "--replication", replication, "--output-root", raw), case_log)
                run_logged((executable, "--fixture-root", raw, "--output-dir", temporary, "--scenario-id", scenario, "--replication", replication, "--worker-devices", args.worker_devices), case_log, native_env)
                if not valid_native_estimates(temporary / "native_estimates.csv"):
                    raise RuntimeError(f"invalid native output for {scenario}/{replication}")
                if native.exists():
                    raise RuntimeError(f"refusing to overwrite uncommitted native output for {scenario}/{replication}")
                os.replace(temporary, native)
                atomic_json(complete, {"scenario_id": scenario, "replication": replication, "status": "OK", "shard": f"{args.case_remainder}/{args.case_modulus}"})
                launched += 1
            except Exception as error:
                atomic_json(case / "FAILED.json", {"scenario_id": scenario, "replication": replication, "error": str(error)})
                raise
            finally:
                os.rmdir(lock)
    with launcher_log.open("a") as stream: stream.write(f"completed launched={launched} skipped={skipped}\n")
    if args.no_aggregate: return
    aggregate = root / "sync_gpu/scripts/aggregate_figure1_native.py"
    run_logged((sys.executable, aggregate, "--cases-root", cases, "--output-dir", args.output_root, "--scenario-start", args.scenario_start, "--scenario-end", args.scenario_end, "--rep-start", args.rep_start, "--rep-end", args.rep_end), launcher_log)
    if args.plot:
        plot = root / "sync_gpu/scripts/plot_figure1_native.py"
        run_logged((sys.executable, plot, "--estimates", args.output_root / "native_estimates.csv", "--png", args.output_root / "figure1_cpp_gpu_native.png", "--pdf", args.output_root / "figure1_cpp_gpu_native.pdf"), launcher_log)


if __name__ == "__main__":
    main()
