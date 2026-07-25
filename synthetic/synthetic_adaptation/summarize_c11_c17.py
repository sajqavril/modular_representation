#!/usr/bin/env python3
"""Aggregate C11--C17 runs and render a compact, dependency-free SVG figure."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
METRICS = ("gate_acc", "plasticity_mcc", "old_reference_MCC", "post_adaptation_top_k_MCC", "forgetting")


def mean_std(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)), "n": int(np.sum(~np.isnan(a)))}


def svg(summary: Dict[str, Dict[str, Dict[str, float]]], path: Path) -> None:
    cases = list(summary)
    width, height, left, top = 1120, 620, 95, 65
    panel_w, panel_h = 470, 205
    panels = [("Gate accuracy", "gate_acc", 0.0, 1.05, left, top),
              ("Plasticity (target MCC)", "plasticity_mcc", 0.0, 1.05, 555, top),
              ("Old-data MCC decomposition", "old_reference_MCC", 0.0, 1.05, left, 340),
              ("Forgetting (reference − post)", "forgetting", -0.05, 0.15, 555, 340)]
    elements = ["<svg xmlns='http://www.w3.org/2000/svg' width='1120' height='620' viewBox='0 0 1120 620'>",
                "<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:19px;font-weight:bold}.label{font-size:12px}.small{font-size:10px}</style>",
                "<rect width='100%' height='100%' fill='white'/>",
                "<text x='560' y='30' text-anchor='middle' class='title'>C11–C17: leakage-free gate, 3 data seeds × 5 training seeds</text>"]
    colors = {"gate_acc": "#2a6fbb", "plasticity_mcc": "#d97706", "old_reference_MCC": "#355c7d", "forgetting": "#c2410c"}
    for title, metric, lo, hi, x0, y0 in panels:
        elements += [f"<text x='{x0}' y='{y0-17}' class='label'>{title}</text>",
                     f"<line x1='{x0}' y1='{y0+panel_h}' x2='{x0+panel_w}' y2='{y0+panel_h}' stroke='#555'/>",
                     f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y0+panel_h}' stroke='#555'/>"]
        for tick in np.linspace(lo, hi, 4):
            y = y0 + panel_h * (1 - (tick - lo) / (hi - lo))
            elements += [f"<line x1='{x0}' y1='{y:.1f}' x2='{x0+panel_w}' y2='{y:.1f}' stroke='#e5e7eb'/>",
                         f"<text x='{x0-7}' y='{y+4:.1f}' text-anchor='end' class='small'>{tick:.2f}</text>"]
        step = panel_w / len(cases)
        for i, case in enumerate(cases):
            v = summary[case][metric]
            mean, std = v["mean"], v["std"]
            cx = x0 + step * (i + 0.5)
            y = y0 + panel_h * (1 - (mean - lo) / (hi - lo))
            ylo = y0 + panel_h * (1 - (max(lo, mean - std) - lo) / (hi - lo))
            yhi = y0 + panel_h * (1 - (min(hi, mean + std) - lo) / (hi - lo))
            elements += [f"<line x1='{cx:.1f}' y1='{ylo:.1f}' x2='{cx:.1f}' y2='{yhi:.1f}' stroke='{colors[metric]}' stroke-width='2'/>",
                         f"<circle cx='{cx:.1f}' cy='{y:.1f}' r='4' fill='{colors[metric]}'/>",
                         f"<text x='{cx:.1f}' y='{y0+panel_h+16}' text-anchor='middle' class='small'>{case[:3]}</text>"]
        if metric == "old_reference_MCC":
            # Overlay post-adaptation term to show the exact decomposition.
            for i, case in enumerate(cases):
                v = summary[case]["post_adaptation_top_k_MCC"]
                cx = x0 + step * (i + 0.5) + 8
                y = y0 + panel_h * (1 - (v["mean"] - lo) / (hi - lo))
                elements.append(f"<rect x='{cx-3:.1f}' y='{y-3:.1f}' width='6' height='6' fill='#16a34a'/>")
            elements.append(f"<text x='{x0+panel_w-2}' y='{y0+13}' text-anchor='end' class='small'>● reference  ■ post-adaptation top-k</text>")
    elements.append("</svg>")
    path.write_text("\n".join(elements))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=HERE / "results" / "c11_c17")
    args = p.parse_args()
    with (args.out_dir / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in METRICS:
            grouped[row["case"]][metric].append(float(row[metric]))
    summary = {case: {metric: mean_std(values) for metric, values in stats.items()}
               for case, stats in grouped.items()}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case"] + [item for metric in METRICS for item in (f"{metric}_mean", f"{metric}_std", f"{metric}_n")])
        for case, stats in summary.items():
            writer.writerow([case] + [stats[m][k] for m in METRICS for k in ("mean", "std", "n")])
    svg(summary, args.out_dir / "c11_c17_metrics.svg")
    print("case | gate acc | plasticity | reference MCC | post-adaptation top-k MCC | forgetting")
    for case, stats in summary.items():
        print(f"{case} | {stats['gate_acc']['mean']:.3f}±{stats['gate_acc']['std']:.3f} | "
              f"{stats['plasticity_mcc']['mean']:.3f}±{stats['plasticity_mcc']['std']:.3f} | "
              f"{stats['old_reference_MCC']['mean']:.3f}±{stats['old_reference_MCC']['std']:.3f} | "
              f"{stats['post_adaptation_top_k_MCC']['mean']:.3f}±{stats['post_adaptation_top_k_MCC']['std']:.3f} | "
              f"{stats['forgetting']['mean']:.6f}±{stats['forgetting']['std']:.6f}")


if __name__ == "__main__":
    main()
