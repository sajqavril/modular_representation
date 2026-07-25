#!/usr/bin/env python3
"""
merge_sweep.py — Merge sweep results from per-seed directories into one CSV,
then rank configs per (transition, method) by composite score.

Usage:
  python merge_sweep.py --sweep_dir results/sweep --out results/sweep/all_metrics.csv
"""
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", default="results/sweep")
    p.add_argument("--out", default="results/sweep/all_metrics.csv")
    args = p.parse_args()

    dfs = []
    for csv_path in sorted(Path(args.sweep_dir).glob("metrics_seed*_case*.csv")):
        df = pd.read_csv(csv_path)
        dfs.append(df)
        print(f"  Loaded {len(df)} rows from {csv_path}")

    if not dfs:
        print("No seed*/metrics.csv found. Check --sweep_dir.")
        return

    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(args.out, index=False)
    print(f"\nMerged {len(all_df)} rows -> {args.out}")
    print(f"Transitions: {sorted(all_df.transition.unique())}")
    print(f"Methods: {sorted(all_df.method.unique())}")
    print(f"Seeds: {sorted(all_df.seed.unique())}")

    # Rank: for each (transition, method), find best config by target_mcc mean
    all_df["forgetting_clip"] = all_df["forgetting"].clip(-0.5, 0.5).fillna(0)
    all_df["score"] = (
        all_df["target_mcc"].fillna(0)
        - all_df["forgetting_clip"]
        + 0.5 * all_df["gate_acc"].fillna(0)
    )

    best = (all_df.groupby(["transition", "method", "config_id", "config_tag"])
            .agg(
                target_mcc_mean=("target_mcc","mean"),
                target_mcc_std=("target_mcc","std"),
                old_after_mean=("old_after_mcc","mean"),
                forgetting_mean=("forgetting","mean"),
                gate_acc_mean=("gate_acc","mean"),
                score_mean=("score","mean"),
                n_seeds=("seed","nunique"),
            )
            .reset_index()
            .sort_values(["transition","method","score_mean"], ascending=[True,True,False]))

    best_path = Path(args.out).parent / "best_per_case_method.csv"
    best.to_csv(best_path, index=False)
    print(f"\nBest config per (transition, method) -> {best_path}")

    # Print summary table
    top1 = best.groupby(["transition","method"]).first().reset_index()
    print("\n── Best target_mcc per (transition, method) ──")
    pivot = top1.pivot(index="transition", columns="method", values="target_mcc_mean")
    print(pivot.round(3).to_string())

if __name__ == "__main__":
    main()