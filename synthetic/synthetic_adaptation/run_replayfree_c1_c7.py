#!/usr/bin/env python3
"""Validate the replay-free gate on original synthetic C1--C7.

Stage 1 sees old observations and returns ``ReplayFreeOldModelState``.  The
stage-2 adaptation function accepts only that state and ``target_X``.  Old
data and ground-truth metadata enter again only after fitting, inside the
evaluation function that computes MCC and forgetting.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "synthetic_adaptation"
sys.path.insert(0, str(SOURCE))

from more_core import (  # noqa: E402
    ALL_TRANSITIONS, CoordBasis, TrainConfig, best_mcc,
    estimate_Z, expand_basis, fit_pca_basis,
    gate_accuracy, generate_data, set_seed, train_more,
)
from rank_aware_gate import decision_metadata, install_fixed_gate  # noqa: E402
from replayfree_gate import (  # noqa: E402
    ReplayFreeOldModelState, fit_old_model_state, initialize_new_encoder_target_ica,
    replayfree_target_gate, state_metadata,
)
from run_c11_c17 import DeepMoRe  # noqa: E402


def basis_from_state(state: ReplayFreeOldModelState) -> CoordBasis:
    return CoordBasis(V=state.basis_V, mean=state.basis_mean)


@torch.no_grad()
def estimate_all_modules(model: DeepMoRe, X_old: torch.Tensor, X_new: torch.Tensor,
                         device: torch.device) -> torch.Tensor:
    """Retention evaluation only: all learned modules available to old input."""
    old_logits = model.gate_old.logits.detach().clone()
    new_logits = model.gate_new.logits.detach().clone() if model.gate_new else None
    try:
        model.gate_old.logits.data.fill_(12.0)
        if model.gate_new is not None:
            model.gate_new.logits.data.fill_(12.0)
        return estimate_Z(model, X_old, X_new, device)
    finally:
        model.gate_old.logits.data.copy_(old_logits)
        if model.gate_new is not None and new_logits is not None:
            model.gate_new.logits.data.copy_(new_logits)


def has_trainable_parameters(model: torch.nn.Module) -> bool:
    return any(p.requires_grad for p in model.parameters())


def adapt_without_replay(old_model: ReplayFreeOldModelState, target_X: torch.Tensor,
                         args: argparse.Namespace, device: torch.device,
                         target_ica_seed: int) -> Tuple[DeepMoRe, Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deployment stage: no old samples or old-task metadata are in its API."""
    basis_old = basis_from_state(old_model)
    decision = replayfree_target_gate(
        old_model, target_X, rank_rtol=args.rank_rtol, target_ica_seed=target_ica_seed,
        target_split_consensus=not args.no_target_split_consensus,
    )
    n_new = int(decision.g_new.numel())
    _, V_new = expand_basis(target_X, basis_old, n_new)
    X_target_old = basis_old.project(target_X)
    X_target_new = ((target_X - basis_old.mean) @ V_new if n_new else
                    torch.zeros(target_X.shape[0], 2, 0))
    # DeepMoRe has a block-coordinate log-determinant and therefore supports
    # every rank-derived new-coordinate width, including low-sample edge cases.
    model = DeepMoRe(old_model.n_old, n_new, old_model.n_old, n_new,
                     old_model.F_old, old_model.B_old)
    install_fixed_gate(model, decision)
    if n_new:
        initialize_new_encoder_target_ica(model.F_new.data, X_target_new, target_ica_seed + 10_000)
    cfg = TrainConfig(
        name="replayfree_fixed_gate", use_align_T=False, freeze_B_old=True,
        use_old_gates=True, use_new_gates=True, epochs=args.epochs, lr=args.lr,
        tau_start=args.tau_start, tau_end=args.tau_end, warmup_frac=0.0,
        w_sp_old=0.0, w_sp_new=0.0, w_B_sp=args.w_B_sp,
        w_gate_binary=0.0, gate_lr_factor=0.0,
    )
    train_time = 0.0
    if has_trainable_parameters(model):
        train_time = float(train_more(model, X_target_old, X_target_new, cfg, device)["train_time_sec"])
    return (model, decision_metadata(decision) | {"train_time_sec": train_time},
            X_target_old, X_target_new, V_new)


def evaluate_case(model: DeepMoRe, decision_meta: Dict[str, Any], V_new: torch.Tensor,
                  old_eval: Dict[str, torch.Tensor], target_eval: Dict[str, torch.Tensor],
                  old_model: ReplayFreeOldModelState, old_active: Sequence[int],
                  target_active: Sequence[int], X_target_old: torch.Tensor,
                  X_target_new: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    """Evaluation boundary, kept separate from ``adapt_without_replay``."""
    basis_old = basis_from_state(old_model)
    Z_target = estimate_Z(model, X_target_old, X_target_new, device)
    target_mcc, _ = best_mcc(Z_target, target_eval["Z"], target_active)
    n_new = int(len(decision_meta["g_new"]))
    new_truth = [i for i in target_active if i not in old_active]
    new_mcc = float("nan")
    if n_new and new_truth:
        new_mcc, _ = best_mcc(Z_target[:, :, old_model.n_old:], target_eval["Z"], new_truth)

    X_old_pca = basis_old.project(old_eval["X"])
    Z_old_ref = X_old_pca @ old_model.F_old
    old_reference_mcc, pairs = best_mcc(Z_old_ref, old_eval["Z"], old_active)
    old_map = {module: truth for module, truth, _ in pairs}
    X_old_new = ((old_eval["X"] - basis_old.mean) @ V_new if n_new else
                 torch.zeros(old_eval["X"].shape[0], 2, 0))
    Z_old_all = estimate_all_modules(model, X_old_pca, X_old_new, device)
    post_mcc, _ = best_mcc(Z_old_all, old_eval["Z"], old_active)
    Z_old_selected = estimate_Z(model, X_old_pca, X_old_new, device)
    selected_mcc, _ = best_mcc(Z_old_selected, old_eval["Z"], old_active)
    gates = torch.tensor(decision_meta["g_old"])
    stats = gate_accuracy(gates, old_map, target_active)
    return {
        "target_mcc": target_mcc, "new_mcc": new_mcc,
        "old_reference_MCC": old_reference_mcc,
        "post_adaptation_top_k_MCC": post_mcc,
        "forgetting": max(0.0, old_reference_mcc - post_mcc),
        "selected_branch_old_MCC_aux": selected_mcc,
        "selected_branch_forgetting_aux": max(0.0, old_reference_mcc - selected_mcc),
        "gate_acc": stats["gate_acc"], **stats,
    }


def one_run(case_idx: int, data_seed: int, train_seed: int, args: argparse.Namespace,
            device: torch.device) -> Dict[str, Any]:
    name, old_name, target_name, old_active, target_active, old_dim, _, _, _, _ = ALL_TRANSITIONS[case_idx]
    # Stage 1: old task is visible only here.
    data = generate_data(args.T, args.obs_dim, args.noise_scale, data_seed)
    old_stage1 = data.scenarios[old_name]
    old_basis = fit_pca_basis(old_stage1["X"], old_dim)
    old_model = fit_old_model_state(old_stage1["X"], old_basis, old_dim,
                                    random_state=50_000 + data_seed,
                                    persistence_threshold=args.persistence_threshold)

    # Stage 2 deployment input: model state plus new observations only.
    target_X = data.scenarios[target_name]["X"]
    set_seed(train_seed)
    model, decision_meta, X_target_old, X_target_new, V_new = adapt_without_replay(
        old_model, target_X, args, device, target_ica_seed=60_000 + data_seed)

    # Post-hoc evaluation boundary. This is the first point at which labels or
    # old samples below are read again.
    result = evaluate_case(model, decision_meta, V_new, data.scenarios[old_name],
                           data.scenarios[target_name], old_model, old_active,
                           target_active, X_target_old, X_target_new, device)
    return {
        "case": name, "data_seed": data_seed, "train_seed": train_seed,
        **result, **decision_meta, **state_metadata(old_model),
    }


def write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=HERE / "results" / "replayfree_c1_c7")
    p.add_argument("--data_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--train_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--cases", type=int, nargs="+", default=list(range(7)))
    p.add_argument("--T", type=int, default=5_000)
    p.add_argument("--obs_dim", type=int, default=8)
    p.add_argument("--noise_scale", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--tau_start", type=float, default=2.0)
    p.add_argument("--tau_end", type=float, default=0.3)
    p.add_argument("--w_B_sp", type=float, default=0.01)
    p.add_argument("--rank_rtol", type=float, default=1e-3)
    p.add_argument("--persistence_threshold", type=float, default=0.90)
    p.add_argument("--no_target_split_consensus", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if set(args.cases) - set(range(7)):
        p.error("cases must be in 0..6")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args) | {"out_dir": str(args.out_dir)}, indent=2, default=str))
    metrics_path = args.out_dir / "metrics.csv"
    rows = read_csv(metrics_path) if args.resume else []
    completed = {(r["case"], int(r["data_seed"]), int(r["train_seed"])) for r in rows}
    for data_seed in args.data_seeds:
        for train_seed in args.train_seeds:
            for case in args.cases:
                name = ALL_TRANSITIONS[case][0]
                key = (name, data_seed, train_seed)
                if key in completed:
                    print(f"skip completed {name} data={data_seed} train={train_seed}", flush=True)
                    continue
                row = one_run(case, data_seed, train_seed, args, torch.device(args.device))
                rows.append(row)
                completed.add(key)
                write_csv(rows, metrics_path)
                print(f"{row['case']} data={data_seed} train={train_seed}: "
                      f"gate={row['gate_acc']:.3f} plasticity={row['target_mcc']:.3f} "
                      f"forgetting={row['forgetting']:.6f}", flush=True)
    print(f"completed {len(rows)} fits", flush=True)


if __name__ == "__main__":
    main()
