#!/usr/bin/env python3
"""Plot native-only Figure-1 final alpha/beta estimates from an aggregate CSV."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt

SCENARIOS = [(0.5,.033),(0.5,.1),(0.5,.234),(1.5,.037),(1.5,.11),(1.5,.256),(2.5,.038),(2.5,.113),(2.5,.264)]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--estimates", type=Path, required=True); parser.add_argument("--png", type=Path, required=True); parser.add_argument("--pdf", type=Path, required=True); args = parser.parse_args()
    with args.estimates.open(newline="") as stream: rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), constrained_layout=True)
    for scenario, (nu, beta_truth) in enumerate(SCENARIOS):
        subset = [row for row in rows if int(row["scenario_id"]) == scenario]
        for col, (parameter, truth) in enumerate((("alpha", 1.0), ("beta", beta_truth))):
            axis = axes[scenario // 3, (scenario % 3) * 2 + col]
            values = [[float(row[parameter]) for row in subset if row["method"] == method] for method in ("IND", "P")]
            if any(values): axis.boxplot(values, labels=["IND", "P"])
            axis.axhline(truth, color="red", linewidth=1)
            axis.set_title(f"nu={nu}, beta={beta_truth:g}: {parameter}")
    args.png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.png, dpi=180); fig.savefig(args.pdf); print(args.png); print(args.pdf)


if __name__ == "__main__":
    main()
