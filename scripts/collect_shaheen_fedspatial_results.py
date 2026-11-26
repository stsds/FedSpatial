#!/usr/bin/env python3
"""Collect one-iteration FedSpatial production artifacts into one CSV."""
import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, List


PMPI = re.compile(
    r"PMPI_PAYLOAD point_to_point_bytes=(\d+) collective_min_bytes=(\d+) total_min_bytes=(\d+)"
)


def key_values(path: Path) -> Dict[str, str]:
    result = {}  # type: Dict[str, str]
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fields = [
        "scaling", "nodes", "nodes_per_worker", "workers", "m", "global_n",
        "local_n", "replica", "status", "seconds", "mu_sigma_seconds",
        "gamma_seconds", "delta_theta_seconds", "logical_bytes",
        "pmpi_point_to_point_bytes", "pmpi_collective_min_bytes",
        "pmpi_total_min_bytes", "peak_rank_rss_kib", "sum_rank_rss_kib",
        "delta", "alpha", "range", "accepted_step", "halving_count",
        "fixture_hash", "job_id", "result_dir",
    ]
    rows = []  # type: List[Dict[str, Any]]
    with args.submission_manifest.open(newline="") as stream:
        configurations = list(csv.DictReader(stream, delimiter="\t"))
    for config in configurations:
        nodes = int(config["nodes"])
        per_worker = int(config["nodes_per_worker"])
        workers = int(config["workers"])
        global_n = int(config["global_n"])
        local_n = global_n // workers
        capacity = per_worker == 4 and local_n > 46340
        for replica in range(3):
            directory = (
                args.output_root / config["scaling"] / f"{nodes}nodes"
                / f"{per_worker}nodes-per-worker" / f"m{config['m']}"
                / f"replica-{replica}"
            )
            row = {  # type: Dict[str, Any]
                **{name: config[name] for name in
                   ("scaling", "nodes", "nodes_per_worker", "workers", "m", "global_n")},
                "local_n": local_n,
                "replica": replica,
                "status": "INFEASIBLE_MPI_COUNT" if capacity else "PENDING",
                "result_dir": str(directory),
            }
            manifest = key_values(directory / "manifest.txt")
            row["fixture_hash"] = manifest.get("fixture_hash", "")
            row["job_id"] = manifest.get("job_id", "")
            timing_path = directory / "timing.csv"
            run_path = directory / "run.log"
            if not capacity and timing_path.exists() and run_path.exists():
                log = run_path.read_text(errors="replace")
                if "SYNCHRONOUS_TRAJECTORY_STATUS=PASS" in log:
                    with timing_path.open(newline="") as stream:
                        timing = list(csv.DictReader(stream))
                    if len(timing) == 1:
                        row.update(timing[0])
                        row["logical_bytes"] = timing[0]["logical_communication_bytes"]
                        match = PMPI.search(log)
                        if match:
                            row["pmpi_point_to_point_bytes"] = match.group(1)
                            row["pmpi_collective_min_bytes"] = match.group(2)
                            row["pmpi_total_min_bytes"] = match.group(3)
                        rss_path = directory / "peak_rss.csv"
                        if rss_path.exists():
                            with rss_path.open(newline="") as stream:
                                rss = [int(x["peak_rss_kib"]) for x in csv.DictReader(stream)]
                            row["peak_rank_rss_kib"] = max(rss)
                            row["sum_rank_rss_kib"] = sum(rss)
                        checkpoints = directory / "checkpoints.csv"
                        if checkpoints.exists():
                            with checkpoints.open(newline="") as stream:
                                final = list(csv.DictReader(stream))[-1]
                            for name in ("delta", "alpha", "range", "accepted_step", "halving_count"):
                                row[name] = final[name]
                        row["status"] = "PASS"
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    counts = {}  # type: Dict[str, int]
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    print(" ".join(f"{key}={value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()
