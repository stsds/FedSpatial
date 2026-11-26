#!/usr/bin/env python3
"""Validate and aggregate fixture-backed native Figure-1 case outputs."""
import argparse
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario-start", type=int, required=True)
    parser.add_argument("--scenario-end", type=int, required=True)
    parser.add_argument("--rep-start", type=int, required=True)
    parser.add_argument("--rep-end", type=int, required=True)
    args = parser.parse_args()
    expected = [(s, r) for s in range(args.scenario_start, args.scenario_end + 1)
                for r in range(args.rep_start, args.rep_end + 1)]
    rows, missing, failures = [], [], []
    for scenario, replication in expected:
        estimates = args.cases_root / f"scenario_{scenario:02d}" / f"rep_{replication:03d}" / "native" / "native_estimates.csv"
        if not estimates.exists():
            missing.append([scenario, replication])
            continue
        with estimates.open(newline="") as stream:
            case_rows = list(csv.DictReader(stream))
        if len(case_rows) != 2 or {row.get("method") for row in case_rows} != {"P", "IND"}:
            failures.append({"scenario": scenario, "replication": replication, "reason": "expected exactly P and IND rows"})
            continue
        for row in case_rows:
            try:
                values = [float(row[name]) for name in ("gamma0", "gamma1", "gamma2", "gamma3", "gamma4", "delta", "alpha", "beta", "runtime_seconds")]
            except (KeyError, ValueError) as error:
                failures.append({"scenario": scenario, "replication": replication, "reason": str(error)})
                continue
            if row.get("status") != "OK" or not all(math.isfinite(value) for value in values):
                failures.append({"scenario": scenario, "replication": replication, "reason": "nonfinite or non-OK result"})
                continue
            rows.append({"scenario_id": scenario, "replication": replication, **row})
    keys = [(row["scenario_id"], row["replication"], row["method"]) for row in rows]
    if len(keys) != len(set(keys)):
        failures.append({"reason": "duplicate scenario/replication/method key"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete = not missing and not failures and len(rows) == 2 * len(expected)
    report = {"expected_cases": len(expected), "complete_cases": len(rows) // 2,
              "expected_estimates": 2 * len(expected), "estimates": len(rows),
              "missing": missing, "failures": failures, "status": "PASS" if complete else "FAIL"}
    (args.output_dir / "completeness.json").write_text(json.dumps(report, indent=2) + "\n")
    if rows:
        with (args.output_dir / "native_estimates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
