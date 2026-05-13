#!/usr/bin/env python3
"""
more_baselines.py — Baselines and ablations for MoRe across the 6 data2 scenarios.

Methods evaluated (S-RQ2: Plasticity and Stability):
  scratch          Train from scratch on target data (no reuse)
  finetune         Fine-tune old encoder on target data
  expansion_only   Freeze old encoder, learn new modules only (no selection)
  MoRe_no_T        Full MoRe but without alignment matrix T (ablation for stability)
  MoRe             Full proposed method (uniform gate init for training stability)

Note on the "MoRe" variant reported in the paper:
  We report the variant with uniform gate logit initialization (gate_init_logit=0.0)
  rather than the analytic-score-based initialization. This is because the analytic
  initialization (using softmax of H1/H3-based scores) introduces training instability
  in early epochs that degrades final results on some transitions. The analytic score
  remains useful as a *diagnostic* (printed for inspection), but the final adaptation
  uses uniform gate init and lets the data loss drive selection through the gates.

Methods evaluated (S-RQ3: Gate decision accuracy):
  MoRe              Full method (both sparsity penalties on)
  MoRe_no_sp_old    MoRe without L0 penalty on old gates (w_sp_old = 0)
  MoRe_no_sp_new    MoRe without L0 penalty on new gates (w_sp_new = 0)

Cases: 6 transitions (case7_z2_only is excluded).

Usage:
  python more_baselines.py --out_dir results/baselines --T 50000 --epochs 5000 --seeds 0 1 2
  # Quick test:
  python more_baselines.py --out_dir results/baselines --T 10000 --epochs 1000 --seeds 0
"""

from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path
from typing import Dict, List, Any, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

from more_core import (
    set_seed, ensure_dir, to_np, generate_data, ALL_TRANSITIONS,
    fit_pca_basis, expand_basis, compute_self_consistency,
    analytic_gate_init_logits,
    MoRe, TrainConfig, train_more, estimate_Z, best_mcc, gate_accuracy,
    CoordBasis,
)


# =============================================================================
# Cases reported in the paper: 6 transitions (case7_z2_only excluded).
# =============================================================================
REPORTED_CASES = [0, 1, 2, 3, 4, 5]   # indices into ALL_TRANSITIONS


# =============================================================================
# Old model training (scratch, no gates)
# =============================================================================

class OldModel(nn.Module):
    """Simple linear encoder + dynamics, trained on data1."""
    def __init__(self, coord_dim: int, m: int):
        super().__init__()
        self.F = nn.Parameter(0.05 * torch.randn(coord_dim, m))
        self.B = nn.Parameter(torch.eye(m) * 0.8)

    def forward(self, X_proj: torch.Tensor) -> Dict[str, Any]:
        Z = X_proj @ self.F
        E = Z[:, 1, :] - Z[:, 0, :] @ self.B.T
        gram = self.F.T @ self.F + 1e-6 * torch.eye(self.F.shape[1], device=self.F.device)
        loss = E.abs().mean() - 0.5 * torch.linalg.slogdet(gram)[1]
        return {"loss": loss, "Z": Z, "F": self.F, "B": self.B}


def train_old_model(X_proj: torch.Tensor, m: int, lr: float,
                    epochs: int, device: torch.device) -> Tuple[OldModel, Dict]:
    X_proj = X_proj.to(device)
    model = OldModel(X_proj.shape[-1], m).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        out = model(X_proj)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(X_proj)
    return model, {"F": out["F"].cpu(), "B": out["B"].cpu(),
                   "Z": out["Z"].cpu()}


# =============================================================================
# Scratch and finetune baselines (don't use MoRe model)
# =============================================================================

def run_scratch(X_target: torch.Tensor, m: int, lr: float,
                epochs: int, device: torch.device, coord_dim: int) -> torch.Tensor:
    basis = fit_pca_basis(X_target, coord_dim)
    X_proj = basis.project(X_target).to(device)
    model, info = train_old_model(X_proj, m, lr, epochs, device)
    return info["Z"]


def run_finetune(X_target: torch.Tensor, init_F: torch.Tensor, init_B: torch.Tensor,
                 coord_dim_old: int, lr: float, epochs: int, device: torch.device,
                 basis_old: CoordBasis) -> Tuple[torch.Tensor, OldModel]:
    """Returns (Z_hat on target, trained model) so caller can eval on old data."""
    X_proj = basis_old.project(X_target).to(device)
    m = init_F.shape[1]
    model = OldModel(coord_dim_old, m).to(device)
    model.F.data = init_F.to(device).clone()
    model.B.data = init_B.to(device).clone()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        out = model(X_proj)
        opt.zero_grad(set_to_none=True); out["loss"].backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(X_proj)
    return out["Z"].cpu(), model


# =============================================================================
# MoRe configurations reported in the paper
# =============================================================================

def build_more_configs(args: argparse.Namespace) -> List[Tuple[str, TrainConfig]]:
    """
    Returns the MoRe variants reported in the paper:
      - MoRe_no_T : ablation for plasticity/stability (S-RQ2)
      - MoRe      : full method (renamed from previous MoRe_no_analytic for clarity)
      - MoRe_no_sp_old : ablation for gate accuracy (S-RQ3)
      - MoRe_no_sp_new : ablation for gate accuracy (S-RQ3)

    Note: All variants use uniform gate logit initialization (use_analytic_init=False,
    gate_init_logit=0.0). The analytic self-consistency score is still computed and
    saved for diagnostic purposes, but is not used to initialize the gate logits in
    the reported MoRe variant due to training-stability issues with the analytic init.
    """
    base = dict(
        epochs=args.epochs, lr=args.lr,
        tau_start=args.tau_start, tau_end=args.tau_end,
        warmup_frac=args.warmup_frac,
        w_data=args.w_data, w_align=args.w_align,
        w_B_sp=args.w_B_sp, w_gate_binary=args.w_gate_binary,
        analytic_gate_C=args.analytic_gate_C,
        gate_lr_factor=args.gate_lr_factor,
    )
    cfgs = [
        # S-RQ2 ablation: MoRe without alignment matrix T.
        ("MoRe_no_T", TrainConfig(
            name="MoRe_no_T", use_align_T=False, freeze_B_old=True,
            use_old_gates=True, use_new_gates=True, use_analytic_init=False,
            freeze_gates_warmup=False, gate_init_logit=0.0,
            w_sp_old=args.w_sp_old, w_sp_new=args.w_sp_new, **base)),
        # Full proposed method.
        # Uniform gate init: analytic score is computed (for inspection) but not used
        # as logit init, because the analytic init introduces training instability.
        ("MoRe", TrainConfig(
            name="MoRe", use_align_T=True, freeze_B_old=True,
            use_old_gates=True, use_new_gates=True, use_analytic_init=False,
            freeze_gates_warmup=False, gate_init_logit=0.0,
            w_sp_old=args.w_sp_old, w_sp_new=args.w_sp_new, **base)),
        # S-RQ3 ablation: MoRe without L0 sparsity on old gates.
        ("MoRe_no_sp_old", TrainConfig(
            name="MoRe_no_sp_old", use_align_T=True, freeze_B_old=True,
            use_old_gates=True, use_new_gates=True, use_analytic_init=False,
            freeze_gates_warmup=False, gate_init_logit=0.0,
            w_sp_old=0.0, w_sp_new=args.w_sp_new, **base)),
        # S-RQ3 ablation: MoRe without L0 sparsity on new gates.
        ("MoRe_no_sp_new", TrainConfig(
            name="MoRe_no_sp_new", use_align_T=True, freeze_B_old=True,
            use_old_gates=True, use_new_gates=True, use_analytic_init=False,
            freeze_gates_warmup=False, gate_init_logit=0.0,
            w_sp_old=args.w_sp_old, w_sp_new=0.0, **base)),
    ]
    return cfgs


# =============================================================================
# Main runner: one transition, all methods
# =============================================================================

def run_one_transition(
    tname: str, old_sname: str, target_sname: str,
    old_active: List[int], target_active: List[int],
    old_basis_dim: int, k_extra: int, target_total_dim: int,
    data, args, device: torch.device, seed: int, out_dir: Path,
) -> List[Dict[str, Any]]:
    rows = []
    old_s = data.scenarios[old_sname]
    target_s = data.scenarios[target_sname]

    # --- Train old model on data1 ---
    basis_old = fit_pca_basis(old_s["X"], old_basis_dim)
    X_old_proj = basis_old.project(old_s["X"])
    old_mdl, old_info = train_old_model(X_old_proj, old_basis_dim, args.lr,
                                         args.old_epochs, device)
    F_old = old_info["F"]   # (coord_dim_old, m_old)
    B_old = old_info["B"]   # (m_old, m_old)

    # MCC of old model on old data (reference).
    Z_old_ref = old_info["Z"]
    old_mcc, _ = best_mcc(Z_old_ref, old_s["Z"], old_active)

    # Map old encoder indices to ground-truth factor indices.
    _, pairs = best_mcc(Z_old_ref, old_s["Z"], old_active)
    old_map = {est: gt for est, gt, _ in pairs}

    # --- Expand coordinates ---
    basis_target, V_new = expand_basis(target_s["X"], basis_old, k_extra)
    X_target_old = basis_target.project(target_s["X"])[:, :, :old_basis_dim]
    X_target_new = (
        (target_s["X"] - basis_old.mean) @ V_new
        if k_extra > 0
        else torch.zeros(target_s["X"].shape[0], 2, 0)
    )
    X_old_in_target = basis_target.project(old_s["X"])[:, :, :old_basis_dim]

    # --- Analytic self-consistency score (diagnostic only) ---
    # We compute and save a_old for inspection, but the reported MoRe variant
    # initializes gate logits uniformly (use_analytic_init=False) due to
    # training-stability issues with the analytic initialization.
    a_old, sc_diag = compute_self_consistency(
        X_target_old.to(device), F_old.to(device), B_old.to(device),
        F_ref=F_old.to(device), X_proj_ref=X_old_proj.to(device),
    )
    a_old_cpu = a_old.cpu()
    print(f"  [{tname}] a_old (diagnostic)={a_old_cpu.numpy().round(4).tolist()}", flush=True)
    torch.save({"sc_diag": sc_diag, "a_old": a_old_cpu},
               out_dir / f"seed{seed}_{tname}_sc_diag.pt")

    # --- Scratch baseline ---
    t0 = time.perf_counter()
    Z_hat = run_scratch(target_s["X"], target_total_dim, args.lr,
                        args.epochs, device, old_basis_dim + k_extra)
    elapsed = time.perf_counter() - t0
    t_mcc, _ = best_mcc(Z_hat, target_s["Z"], target_active)
    # Scratch discards old encoder entirely — old knowledge is lost.
    rows.append({
        "seed": seed, "transition": tname, "method": "scratch",
        "target_mcc": t_mcc, "old_mcc_ref": old_mcc,
        "old_after_mcc": float("nan"), "forgetting": float("nan"),
        "gate_acc": float("nan"), "n_active_old": float("nan"),
        "a_old": a_old_cpu.tolist(), "g_old": None, "g_new": None,
        "train_time": elapsed, "old_name": old_sname, "target_name": target_sname,
        "old_active": old_active, "target_active": target_active,
    })

    # --- Expansion only baseline ---
    t0 = time.perf_counter()
    model_exp = MoRe(
        coord_dim_old=old_basis_dim, coord_dim_new=k_extra,
        m_old=old_basis_dim, m_new=k_extra,
        a_old=torch.ones(old_basis_dim),
        init_F=F_old, init_B=B_old,
        use_align_T=False, freeze_B_old=True,
        gate_init_logit=5.0,
    )
    model_exp.gate_old.logits.requires_grad = False
    if model_exp.gate_new is not None:
        model_exp.gate_new.logits.data.fill_(0.0)
    cfg_exp = TrainConfig(
        name="expansion_only", use_old_gates=False, use_new_gates=True,
        use_analytic_init=False, freeze_gates_warmup=False,
        epochs=args.epochs, lr=args.lr,
        tau_start=args.tau_start, tau_end=args.tau_end,
        warmup_frac=0.0, w_sp_old=0.0, w_sp_new=args.w_sp_new,
        w_B_sp=args.w_B_sp, w_align=0.0, w_gate_binary=0.0,
        gate_lr_factor=1.0,
    )
    train_more(model_exp, X_target_old, X_target_new, cfg_exp, device)
    Z_hat = estimate_Z(model_exp, X_target_old, X_target_new, device)
    # Old encoder is frozen — backward compat is exact (forgetting = 0).
    Z_old_exp = estimate_Z(model_exp, X_old_in_target,
                            torch.zeros(X_old_in_target.shape[0], 2, k_extra), device)
    exp_old_after, _ = best_mcc(Z_old_exp, old_s["Z"], old_active)
    t_mcc, _ = best_mcc(Z_hat, target_s["Z"], target_active)
    rows.append({
        "seed": seed, "transition": tname, "method": "expansion_only",
        "target_mcc": t_mcc, "old_mcc_ref": old_mcc,
        "old_after_mcc": exp_old_after,
        "forgetting": old_mcc - exp_old_after,
        "gate_acc": float("nan"), "n_active_old": float("nan"),
        "a_old": a_old_cpu.tolist(), "g_old": None, "g_new": None,
        "train_time": time.perf_counter() - t0,
        "old_name": old_sname, "target_name": target_sname,
        "old_active": old_active, "target_active": target_active,
    })

    # --- Finetune baseline ---
    t0 = time.perf_counter()
    Z_ft, ft_model = run_finetune(target_s["X"], F_old, B_old, old_basis_dim,
                                   args.lr, args.epochs, device, basis_old)
    t_mcc_ft, _ = best_mcc(Z_ft, target_s["Z"], target_active)
    # Eval fine-tuned model on old data (projected via old basis).
    X_old_proj_eval = basis_old.project(old_s["X"]).to(device)
    ft_model.eval()
    with torch.no_grad():
        ft_out = ft_model(X_old_proj_eval)
    Z_ft_old = ft_out["Z"].cpu()
    ft_old_after, _ = best_mcc(Z_ft_old, old_s["Z"], old_active)
    rows.append({
        "seed": seed, "transition": tname, "method": "finetune",
        "target_mcc": t_mcc_ft, "old_mcc_ref": old_mcc,
        "old_after_mcc": ft_old_after,
        "forgetting": old_mcc - ft_old_after,
        "gate_acc": float("nan"), "n_active_old": float("nan"),
        "a_old": a_old_cpu.tolist(), "g_old": None, "g_new": None,
        "train_time": time.perf_counter() - t0,
        "old_name": old_sname, "target_name": target_sname,
        "old_active": old_active, "target_active": target_active,
    })

    # --- MoRe variants (MoRe_no_T, MoRe, MoRe_no_sp_old, MoRe_no_sp_new) ---
    for method_name, cfg in build_more_configs(args):
        print(f"\n  [{tname}] method={method_name}", flush=True)

        # Uniform gate logit initialization. (Analytic init disabled — see header.)
        gate_logits_init = torch.full((old_basis_dim,), cfg.gate_init_logit)

        model = MoRe(
            coord_dim_old=old_basis_dim, coord_dim_new=k_extra,
            m_old=old_basis_dim, m_new=k_extra,
            a_old=a_old_cpu,
            init_F=F_old, init_B=B_old,
            use_align_T=cfg.use_align_T,
            freeze_B_old=cfg.freeze_B_old,
            gate_init_logit=cfg.gate_init_logit,
            gate_logits_override=gate_logits_init,
            gate_logit_new_override=None,
        )
        if not cfg.use_old_gates:
            # Fix all old gates to 1 (no selection).
            model.gate_old.logits.data.fill_(5.0)
            model.gate_old.logits.requires_grad = False

        info = train_more(model, X_target_old, X_target_new, cfg, device)
        Z_hat = estimate_Z(model, X_target_old, X_target_new, device)
        t_mcc, t_pairs = best_mcc(Z_hat, target_s["Z"], target_active)

        # Backward compatibility: eval old model on old data using adapted model.
        Z_old_after = estimate_Z(model, X_old_in_target, torch.zeros(
            X_old_in_target.shape[0], 2, k_extra), device)
        old_after_mcc, _ = best_mcc(Z_old_after, old_s["Z"], old_active)

        g_old = info["out_hard"]["g_old"]
        g_new = info["out_hard"]["g_new"]
        gstats = gate_accuracy(g_old, old_map, target_active)

        rows.append({
            "seed": seed, "transition": tname, "method": method_name,
            "target_mcc": t_mcc, "old_mcc_ref": old_mcc,
            "old_after_mcc": old_after_mcc,
            "forgetting": old_mcc - old_after_mcc,
            "gate_acc": gstats["gate_acc"],
            "n_active_old": gstats["n_active"],
            "a_old": a_old_cpu.tolist(),
            "g_old": to_np(g_old).tolist() if g_old is not None else None,
            "g_new": to_np(g_new).tolist() if g_new is not None else None,
            "train_time": info["train_time_sec"],
            "n_params": info["n_params"],
            "old_name": old_sname, "target_name": target_sname,
            "old_active": old_active, "target_active": target_active,
            **{f"gate_{k}": v for k, v in gstats.items()},
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
    p.add_argument("--out_dir", default="results/baselines")
    p.add_argument("--T", type=int, default=50000)
    p.add_argument("--obs_dim", type=int, default=8)
    p.add_argument("--noise_scale", type=float, default=0.1)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose_every", type=int, default=0)
    # Training
    p.add_argument("--old_epochs", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--tau_start", type=float, default=2.0)
    p.add_argument("--tau_end", type=float, default=0.3)
    p.add_argument("--warmup_frac", type=float, default=0.3)
    # Loss weights
    p.add_argument("--w_data", type=float, default=1.0)
    p.add_argument("--w_align", type=float, default=0.01)
    p.add_argument("--w_sp_old", type=float, default=0.01)
    p.add_argument("--w_sp_new", type=float, default=0.05)
    p.add_argument("--w_B_sp", type=float, default=0.01)
    p.add_argument("--w_gate_binary", type=float, default=0.005)
    p.add_argument("--analytic_gate_C", type=float, default=3.0)
    p.add_argument("--analytic_noise_floor", type=float, default=0.01,
                   help="Diagnostic only; analytic init is disabled in reported MoRe.")
    p.add_argument("--gate_lr_factor", type=float, default=0.1)
    # Transition selection — default is the 6 reported cases (case7_z2_only excluded).
    p.add_argument("--cases", type=int, nargs="*", default=REPORTED_CASES,
                   help=f"Which cases to run (0-indexed). Default: {REPORTED_CASES}.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    device = torch.device(args.device)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    all_rows = []
    for seed in args.seeds:
        print(f"\n{'='*60}\nseed={seed}\n{'='*60}", flush=True)
        set_seed(seed)
        data = generate_data(args.T, args.obs_dim, args.noise_scale, seed)
        torch.save({"mixing_A": data.mixing_A}, out_dir / f"seed{seed}_data.pt")

        for case_idx in args.cases:
            tname, old_sname, target_sname, old_active, target_active, \
                old_basis_dim, k_extra, target_total_dim, _, _ = ALL_TRANSITIONS[case_idx]
            print(f"\n[seed={seed}] Case {case_idx}: {tname}", flush=True)
            rows = run_one_transition(
                tname, old_sname, target_sname, old_active, target_active,
                old_basis_dim, k_extra, target_total_dim,
                data, args, device, seed, out_dir,
            )
            all_rows.extend(rows)
            write_csv(all_rows, out_dir / "metrics.csv")

    # Summary.
    summary = {}
    for tr in sorted(set(r["transition"] for r in all_rows)):
        summary[tr] = {}
        for meth in sorted(set(r["method"] for r in all_rows if r["transition"] == tr)):
            sub = [r for r in all_rows if r["transition"] == tr and r["method"] == meth]
            for metric in ["target_mcc", "old_after_mcc", "forgetting", "gate_acc"]:
                vals = [float(r[metric]) for r in sub
                        if isinstance(r.get(metric), (int, float))
                        and not math.isnan(float(r[metric]))]
                summary[tr].setdefault(meth, {})[metric] = {
                    "mean": float(np.mean(vals)) if vals else None,
                    "std": float(np.std(vals)) if vals else None,
                }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Results: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()