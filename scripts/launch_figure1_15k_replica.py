#!/usr/bin/env python3
"""Detached, watchdog-capped launcher for the 2x15k Figure-1 P target."""
import csv
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def verify_target_fixture(fixture):
    metadata_path = fixture / "metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(f"missing target fixture metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "scenario_id": 4, "replication": 0, "seed": 1, "n": 30000,
        "p": 5, "knots": 2000, "workers": 2,
        "observations_per_worker": 15000,
        "partition_order": "global_then_consecutive",
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items()
                  if metadata.get(key) != value}
    if mismatches:
        raise SystemExit(f"refusing non-target fixture (including 45000-row fixtures): {mismatches}")


def ensure_target_fixture(fixture, generator_log):
    if (fixture / "metadata.json").is_file():
        verify_target_fixture(fixture)
        return
    expected_tail = ("scenario_04", "rep_000")
    if fixture.parent.name != expected_tail[0] or fixture.name != expected_tail[1]:
        raise SystemExit("missing target fixture must be .../scenario_04/rep_000")
    exporter = Path(__file__).resolve().parents[1] / "tests" / "export_figure1_python_fixture.py"
    with generator_log.open("w", buffering=1) as log:
        subprocess.run([sys.executable, str(exporter), "--scenario-id", "4", "--replication", "0",
                        "--n", "30000", "--knots", "2000", "--workers", "2",
                        "--two-worker-15k-target", "--output-root", str(fixture.parent.parent)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    verify_target_fixture(fixture)


def telemetry(stop, path):
    fields = "uuid,pci.bus_id,utilization.gpu,utilization.memory,memory.used,memory.free"
    with path.open("w", buffering=1) as out:
        out.write("timestamp_utc,uuid,pci_bus,utilization_gpu_pct,utilization_memory_pct,memory_used_mib,memory_free_mib\n")
        while not stop.wait(5):
            try:
                text = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True)
                for line in text.splitlines():
                    values = [v.strip() for v in line.split(",")]
                    if len(values) == 6:
                        out.write(stamp() + "," + ",".join(values) + "\n")
            except Exception as error:
                out.write(f"{stamp()},telemetry_error,{type(error).__name__},,,\n")


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit("usage: launcher OUTPUT_DIR EXECUTABLE FIXTURE_DIR [--one-round-authorized]")
    output, executable, fixture = map(Path, sys.argv[1:4])
    one_round = len(sys.argv) == 5
    if one_round and sys.argv[4] != "--one-round-authorized":
        raise SystemExit("unknown fourth argument")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "GENERATING_FIXTURE", "start_utc": stamp()})
    ensure_target_fixture(fixture, output / "fixture_generation.log")
    config = {
        "kind": "authorized_single_timing_replica", "start_utc": stamp(),
        "method": "P_low_rank_synchronous", "rounds": 1 if one_round else 50, "workers": 2,
        "worker_devices": [1, 2], "device0_forbidden": True,
        "n_per_worker": 15000, "total_n": 30000, "m": 2000,
        "q": 5, "precision": "FP64", "scenario": 4, "replication": 0,
        "seed": 1, "nu": 1.5, "delta": .25, "alpha": 1., "beta": .11,
        "fixture": str(fixture), "tiled_path": True, "scalar_fallback": False,
        "watchdog_seconds": 7200 if one_round else 43200,
    }
    write_json(output / "launch_manifest.json", config)
    write_json(output / "preflight_parity.json", {
        "nworker_p_parity": "PASS", "determinism": "BYTE_IDENTICAL",
        "max_abs": 8.881784197001252e-16, "max_rel": 4.770353808768319e-16,
        "note": "No full-size Python reference exists; final errors are recorded when available."
    })
    (output / "stage_events.csv").write_text("raw_log_line\n")
    env = os.environ.copy()
    env.pop("FEDSPATIAL_DISABLE_TILED_SOLVES", None)
    env["FEDSPATIAL_STAGE_CUDA_TIMING"] = "1"
    # Redundant with the in-process guard, but it ensures every child created
    # by the wrapper sees only the two approved physical devices from exec.
    env["CUDA_VISIBLE_DEVICES"] = "GPU-737f3c7a-bf62-20b7-01f8-b3fb1a3c86f4,GPU-a7c40d19-eb27-1d13-257a-5949f98fc947"
    for gate in ("FEDSPATIAL_MIXED_TRACE_TILED_ACTION", "FEDSPATIAL_MIXED_CONTRACTIONS_TILED",
                 "FEDSPATIAL_THETA_FIRST_TILED", "FEDSPATIAL_THETA_HESSIAN_AA_LOG_TILED",
                 "FEDSPATIAL_THETA_HESSIAN_AR_LOG_TILED", "FEDSPATIAL_THETA_HESSIAN_RR_LOG_TILED",
                 "FEDSPATIAL_THETA_HESSIAN_Q_W_TILED"):
        env[gate] = "1"
    config["cuda_visible_devices"] = env["CUDA_VISIBLE_DEVICES"]
    config["optimization_gates"] = {key: env[key] for key in env if key.startswith("FEDSPATIAL_") and key.endswith("TILED")}
    write_json(output / "launch_manifest.json", config)
    command = ["timeout", "--preserve-status", "7200s" if one_round else "43200s", "stdbuf", "-oL", "-eL", str(executable),
               "--fixture-root", str(fixture.parent.parent), "--output-dir", str(output),
               "--scenario-id", "4", "--replication", "0", "--worker-devices", "1,2",
               "--workers", "2", "--rounds", "1" if one_round else "50", "--p-only", "--two-worker-15k-target"]
    if one_round:
        command.append("--one-round-timing")
    (output / "launch_command.txt").write_text(" ".join(command) + "\n")
    stop = threading.Event()
    thread = threading.Thread(target=telemetry, args=(stop, output / "telemetry.csv"), daemon=True)
    with (output / "run.log").open("w", buffering=1) as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        (output / "pid.txt").write_text(str(process.pid) + "\n")
        write_json(output / "status.json", {"status": "RUNNING", "pid": process.pid, "start_utc": stamp()})
        thread.start()
        code = process.wait()
    stop.set(); thread.join(timeout=10)
    status = "PASS" if code == 0 else ("WATCHDOG" if code == 124 else "FAIL")
    write_json(output / "status.json", {"status": status, "pid": process.pid, "exit_code": code, "end_utc": stamp()})
    estimates = output / "native_estimates.csv"
    if estimates.exists():
        rows = list(csv.DictReader(estimates.open()))
        write_json(output / "final_parameters.json", rows)
    # Durable parseable copy of flushed stage boundaries, including cache/tile counters.
    with (output / "stage_events.csv").open("w", buffering=1) as out:
        out.write("raw_log_line\n")
        for line in (output / "run.log").read_text(errors="replace").splitlines():
            if line.startswith(("NWORKER_STAGE", "NWORKER_STAGE_TIMING", "DELTA_THETA_STATE_CACHE")):
                out.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
