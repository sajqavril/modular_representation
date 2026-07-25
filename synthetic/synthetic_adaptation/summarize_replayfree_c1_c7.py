#!/usr/bin/env python3
"""Summarize replay-free C1--C7 metrics from a single long-form CSV."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METRICS = ("gate_acc", "target_mcc", "new_mcc", "old_reference_MCC",
           "post_adaptation_top_k_MCC", "forgetting")


def stats(values):
    v = np.asarray(values, dtype=float)
    if not np.any(~np.isnan(v)):
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
            "n": int(np.sum(~np.isnan(v)))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=HERE / "results" / "replayfree_c1_c7")
    args = p.parse_args()
    with (args.out_dir / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in METRICS:
            grouped[row["case"]][metric].append(float(row[metric]))
    summary = {case: {metric: stats(values) for metric, values in metric_rows.items()}
               for case, metric_rows in grouped.items()}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case"] + [f"{m}_{suffix}" for m in METRICS for suffix in ("mean", "std", "n")])
        for case, result in summary.items():
            writer.writerow([case] + [result[m][key] for m in METRICS for key in ("mean", "std", "n")])
    print("case | gate acc | plasticity | new-factor MCC | old reference | post top-k | forgetting")
    for case, result in summary.items():
        f = lambda metric: f"{result[metric]['mean']:.3f}±{result[metric]['std']:.3f}"
        print(f"{case} | {f('gate_acc')} | {f('target_mcc')} | {f('new_mcc')} | "
              f"{f('old_reference_MCC')} | {f('post_adaptation_top_k_MCC')} | {f('forgetting')}")


if __name__ == "__main__":
    main()
