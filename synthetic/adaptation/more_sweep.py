#!/usr/bin/env python3
"""
more_sweep.py — Hyperparameter sweep for MoRe across the 6 data2 scenarios.

This sweep evaluates the methods reported in the paper:

  S-RQ2 (Plasticity and Stability) baselines and ablations:
    MoRe_no_T        — full MoRe but without alignment matrix T
    MoRe             — full proposed method
                       (uniform gate logit init; the analytic self-consistency
                        score is computed for diagnostic purposes but is not used
                        as the gate-logit initializer in the reported variant
                        because the analytic init introduces training instability)

  S-RQ3 (Gate decision accuracy) ablations:
    MoRe_no_sp_old   — MoRe without L0 sparsity on old gates
    MoRe_no_sp_new   — MoRe without L0 sparsity on new gates

Sweeps over:
  w_sp_old       : L0 penalty on old gates (controls how aggressively to drop)
  w_sp_new       : L0 penalty on new gates (controls expansion conservatism)
  w_align        : alignment matrix T regularization
  w_gate_binary  : binary penalty (complements Hard Concrete discreteness)
  warmup_frac    : fraction of training to freeze gates
  tau_end        : final temperature for Hard Concrete

Outputs:
  metrics_seeds<s>_cases<c>.csv — one row per (seed, config, transition, method)

Cases: 6 transitions (case7_z2_only excluded).

Usage:
  # Full sweep (use --max_configs to limit):
  python more_sweep.py --out_dir results/sweep --seeds 0 1 2 --max_configs 30

  # Quick check:
  python more_sweep.py --out_dir results/sweep_quick --seeds 0 --max_configs 5 \\
      --epochs 1000 --T 10000
"""

from __future__ import annotations
import argparse, csv, json, math, random, time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import torch

from more_core import (
    set_seed, ensure_dir, to_np, generate_data, ALL_TRANSITIONS,
    fit_pca_basis, expand_basis, compute_self_consistency,
    analytic_gate_init_logits,
    MoRe, TrainConfig, train_more, estimate_Z, best_mcc, gate_accuracy,
)
from more_baselines import train_old_model, OldModel, REPORTED_CASES


# =============================================================================
# Grid definition
# =============================================================================

def build_grid(args) -> List[Dict[str, float]]:
    def fl(s): return [float(x) for x in s.split(",") if x.strip()]
    grid = []
    for w_sp_old in fl(args.w_sp_old_grid):
        for w_sp_new in fl(args.w_sp_new_grid):
            for w_align in fl(args.w_align_grid):
                for w_gate_binary in fl(args.w_gate_binary_grid):
                    for warmup in fl(args.warmup_frac_grid):
                        for tau_end in fl(args.tau_end_grid):
                            grid.append({
                                "w_sp_old": w_sp_old,
                                "w_sp_new": w_sp_new,
                                "w_align": w_align,
                                "w_gate_binary": w_gate_binary,
                                "warmup_frac": warmup,
                                "tau_end": tau_end,
                            })
    if args.random_search:
        random.Random(42).shuffle(grid)
    if args.max_configs > 0:
        grid = grid[:args.max_configs]
    return grid


def config_tag(hp: Dict) -> str:
    def f(x): return f"{x:g}".replace(".", "p").replace("-", "m")
    return (f"sp{f(hp['w_sp_old'])}_spn{f(hp['w_sp_new'])}_"
            f"al{f(hp['w_align'])}_bin{f(hp['w_gate_binary'])}_"
            f"w{f(hp['warmup_frac'])}_t{f(hp['tau_end'])}")


# =============================================================================
# Run one (transition, hp config) — evaluates the methods reported in the paper
# =============================================================================

def run_one(
    tname, old_sname, target_sname, old_active, target_active,
    old_basis_dim, k_extra, data, hp, args, device, seed, cfg_id, tag, case_idx,
):
    rows = []
    old_s = data.scenarios[old_sname]
    target_s = data.scenarios[target_sname]

    basis_old = fit_pca_basis(old_s["X"], old_basis_dim)
    X_old_proj = basis_old.project(old_s["X"])
    old_mdl, old_info = train_old_model(X_old_proj, old_basis_dim, args.lr, args.old_epochs, device)
    F_old, B_old = old_info["F"], old_info["B"]
    old_mcc, _ = best_mcc(old_info["Z"], old_s["Z"], old_active)
    _, pairs = best_mcc(old_info["Z"], old_s["Z"], old_active)
    old_map = {est: gt for est, gt, _ in pairs}

    basis_target, V_new = expand_basis(target_s["X"], basis_old, k_extra)
    X_target_old = basis_target.project(target_s["X"])[:, :, :old_basis_dim]
    X_target_new = (
        (target_s["X"] - basis_old.mean) @ V_new if k_extra > 0
        else torch.zeros(target_s["X"].shape[0], 2, 0)
    )
    X_old_in_target = basis_target.project(old_s["X"])[:, :, :old_basis_dim]

    # Diagnostic only: analytic self-consistency score (not used as gate init
    # in the reported MoRe variant — see header).
    a_old, _ = compute_self_consistency(X_target_old.to(device), F_old.to(device), B_old.to(device))
    a_old_cpu = a_old.cpu()
    print(f"    a_old (diag)={a_old_cpu.numpy().round(4).tolist()}", flush=True)

    # Methods reported in the paper. All use uniform gate logit initialization.
    # (name, use_align_T, w_sp_old, w_sp_new, w_align, w_gate_binary)
    methods = [
        ("MoRe_no_T",      False, hp["w_sp_old"], hp["w_sp_new"], 0.0,           hp["w_gate_binary"]),
        ("MoRe",           True,  hp["w_sp_old"], hp["w_sp_new"], hp["w_align"], hp["w_gate_binary"]),
        ("MoRe_no_sp_old", True,  0.0,            hp["w_sp_new"], hp["w_align"], hp["w_gate_binary"]),
        ("MoRe_no_sp_new", True,  hp["w_sp_old"], 0.0,            hp["w_align"], hp["w_gate_binary"]),
    ]

    for (mname, use_align_T, w_sp_old, w_sp_new, w_align, w_gate_binary) in methods:
        cfg = TrainConfig(
            name=f"{mname}__{tag}", use_align_T=use_align_T, freeze_B_old=True,
            use_old_gates=True, use_new_gates=True,
            use_analytic_init=False, freeze_gates_warmup=False, gate_init_logit=0.0,
            epochs=args.epochs, lr=args.lr, tau_start=args.tau_start, tau_end=hp["tau_end"],
            warmup_frac=hp["warmup_frac"],
            w_data=1.0, w_align=w_align, w_sp_old=w_sp_old, w_sp_new=w_sp_new,
            w_B_sp=args.w_B_sp, w_gate_binary=w_gate_binary,
            analytic_gate_C=args.analytic_gate_C, gate_lr_factor=args.gate_lr_factor,
        )
        model = MoRe(
            coord_dim_old=old_basis_dim, coord_dim_new=k_extra,
            m_old=old_basis_dim, m_new=k_extra,
            a_old=a_old_cpu, init_F=F_old, init_B=B_old,
            use_align_T=use_align_T, freeze_B_old=True,
            gate_init_logit=0.0,
            gate_logits_override=None,
            gate_logit_new_override=None,
        )

        info = train_more(model, X_target_old, X_target_new, cfg, device)
        Z_hat = estimate_Z(model, X_target_old, X_target_new, device)
        t_mcc, _ = best_mcc(Z_hat, target_s["Z"], target_active)
        Z_old_after = estimate_Z(model, X_old_in_target,
                                  torch.zeros(X_old_in_target.shape[0], 2, k_extra), device)
        old_after_mcc, _ = best_mcc(Z_old_after, old_s["Z"], old_active)
        g_old = info["out_hard"]["g_old"]
        g_new = info["out_hard"]["g_new"]
        gstats = gate_accuracy(g_old, old_map, target_active)

        rows.append({
            "seed": seed, "config_id": cfg_id, "config_tag": tag,
            "transition": tname, "case_idx": case_idx,
            "method": mname,
            "old_name": old_sname, "target_name": target_sname,
            "old_active": old_active, "target_active": target_active,
            "target_mcc": t_mcc, "old_mcc_ref": old_mcc,
            "old_after_mcc": old_after_mcc, "forgetting": old_mcc - old_after_mcc,
            "gate_acc": gstats["gate_acc"], "n_active_old": gstats["n_active"],
            "a_old": a_old_cpu.tolist(),
            "g_old": to_np(g_old).tolist() if g_old is not None else None,
            "g_new": to_np(g_new).tolist() if g_new is not None else None,
            "train_time": info["train_time_sec"], "n_params": info["n_params"],
            **gstats, **hp,
        })

    return rows


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows: return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            rr = {}
            for k in keys:
                v = r.get(k)
                rr[k] = json.dumps(v) if isinstance(v, (list, dict)) else (
                    "nan" if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
            w.writerow(rr)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/sweep")
    p.add_argument("--T", type=int, default=50000)
    p.add_argument("--obs_dim", type=int, default=8)
    p.add_argument("--noise_scale", type=float, default=0.1)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--old_epochs", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--tau_start", type=float, default=2.0)
    p.add_argument("--w_B_sp", type=float, default=0.01)
    p.add_argument("--gate_lr_factor", type=float, default=0.1)
    p.add_argument("--analytic_gate_C", type=float, default=3.0)
    # Sweep grids — w_sp_new includes very small values to avoid suppressing new modules.
    p.add_argument("--w_sp_old_grid", default="0.005,0.01,0.05")
    p.add_argument("--w_sp_new_grid", default="0.001,0.005,0.01,0.02,0.05")
    p.add_argument("--w_align_grid", default="0.005,0.01,0.02")
    p.add_argument("--w_gate_binary_grid", default="0.001,0.005,0.01")
    p.add_argument("--warmup_frac_grid", default="0.0,0.2,0.3")
    p.add_argument("--tau_end_grid", default="0.2,0.3,0.5")
    p.add_argument("--max_configs", type=int, default=0, help="0=all")
    p.add_argument("--random_search", action="store_true")
    # Default: 6 reported cases (case7_z2_only excluded).
    p.add_argument("--cases", type=int, nargs="*", default=REPORTED_CASES)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    device = torch.device(args.device)
    grid = build_grid(args)

    # Unique filename per (seeds, cases) — guaranteed no collision across parallel jobs.
    cases_str = "_".join(str(c) for c in sorted(args.cases))
    seeds_str = "_".join(str(s) for s in sorted(args.seeds))
    job_csv = out_dir / f"metrics_seeds{seeds_str}_cases{cases_str}.csv"

    # Remove stale file from previous run so append logic starts clean.
    if job_csv.exists():
        job_csv.unlink()

    with open(out_dir / f"args_seeds{seeds_str}_cases{cases_str}.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    n_methods = 4   # MoRe_no_T, MoRe, MoRe_no_sp_old, MoRe_no_sp_new
    print(f"Output: {job_csv}", flush=True)
    print(f"Grid: {len(grid)} configs x {len(args.seeds)} seeds x "
          f"{len(args.cases)} cases x {n_methods} methods = "
          f"{len(grid)*len(args.seeds)*len(args.cases)*n_methods} total runs", flush=True)

    header_written = False

    for seed in args.seeds:
        print(f"\n{'='*60}\nseed={seed}\n{'='*60}", flush=True)
        set_seed(seed)
        data = generate_data(args.T, args.obs_dim, args.noise_scale, seed)

        for cfg_id, hp in enumerate(grid):
            tag = config_tag(hp)
            print(f"\n  cfg {cfg_id+1}/{len(grid)}: {tag}", flush=True)
            for case_idx in args.cases:
                tname, old_sn, tgt_sn, old_act, tgt_act, \
                    old_bdim, k_extra, total_dim, _, _ = ALL_TRANSITIONS[case_idx]
                rows = []
                try:
                    rows = run_one(
                        tname, old_sn, tgt_sn, old_act, tgt_act,
                        old_bdim, k_extra, data, hp, args, device, seed, cfg_id, tag,
                        case_idx,
                    )
                except Exception as e:
                    import traceback
                    print(f"  ERROR {tname} cfg{cfg_id}: {e}", flush=True)
                    traceback.print_exc()

                # Append rows immediately — never accumulate in memory across seeds/configs.
                if rows:
                    keys = sorted({k for r in rows for k in r})
                    with open(job_csv, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=keys)
                        if not header_written:
                            w.writeheader()
                            header_written = True
                        for r in rows:
                            rr = {}
                            for k in keys:
                                v = r.get(k)
                                rr[k] = json.dumps(v) if isinstance(v, (list, dict)) else (
                                    "nan" if isinstance(v, float) and
                                    (math.isnan(v) or math.isinf(v)) else v)
                            w.writerow(rr)

    print(f"\nDone. Results: {job_csv.resolve()}", flush=True)


if __name__ == "__main__":
    main()