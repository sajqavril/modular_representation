#!/usr/bin/env python3
"""Replay-free C11--C17 experiment: 3 layers x 5 components.

Only ``adapt_without_replay`` is permitted to make a structural/adaptation
decision. Its API contains a saved ``ReplayFreeOldModelState`` and target
observations, never old samples or synthetic labels. Evaluation is explicitly
separate and runs after adaptation has completed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "synthetic_adaptation"
sys.path.insert(0, str(SOURCE))

from more_core import CoordBasis, TrainConfig, estimate_Z, expand_basis, fit_pca_basis, set_seed, train_more  # noqa: E402
from rank_aware_gate import decision_metadata, install_fixed_gate  # noqa: E402
from replayfree_gate import (  # noqa: E402
    ReplayFreeOldModelState, fit_old_model_state, initialize_new_encoder_target_ica,
    replayfree_target_gate, state_metadata,
)
# These are data/model utilities only. The legacy C11 runner itself is not used.
from run_c11_c17 import (  # noqa: E402
    N_PER_LAYER, OLD_FACTORS, SCENARIOS, DeepMoRe, gate_stats,
    generate_three_layer_data, hungarian_mcc,
)


def basis_from_state(state: ReplayFreeOldModelState) -> CoordBasis:
    return CoordBasis(V=state.basis_V, mean=state.basis_mean)


@torch.no_grad()
def estimate_all_modules(model: DeepMoRe, X_old: torch.Tensor, X_new: torch.Tensor,
                         device: torch.device) -> torch.Tensor:
    """Offline retention evaluation; never called by adaptation."""
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
    """Target-only deployment path; no old-task argument exists in this API."""
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
    model = DeepMoRe(N_PER_LAYER, n_new, N_PER_LAYER, n_new, old_model.F_old, old_model.B_old)
    install_fixed_gate(model, decision)
    if n_new:
        initialize_new_encoder_target_ica(model.F_new.data, X_target_new, target_ica_seed + 10_000)
        if args.freeze_new_ica_encoder:
            # ICA already identifies independent target directions at t=0.
            # Keep that source coordinate system fixed while fitting only the
            # target dynamics, rather than allowing an unsupervised temporal
            # loss to rotate the encoder away from its identifiable solution.
            model.F_new.requires_grad_(False)
    cfg = TrainConfig(
        name="replayfree_C11_C17", use_align_T=False, freeze_B_old=True,
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


def evaluate_after_adaptation(model: DeepMoRe, decision: Dict[str, Any], V_new: torch.Tensor,
                              old_eval: Dict[str, torch.Tensor], target_eval: Dict[str, torch.Tensor],
                              old_model: ReplayFreeOldModelState, target_active: Sequence[int],
                              X_target_old: torch.Tensor, X_target_new: torch.Tensor,
                              device: torch.device) -> Dict[str, Any]:
    """Offline metric computation; this is outside the deployment boundary."""
    basis_old = basis_from_state(old_model)
    Z_target = estimate_Z(model, X_target_old, X_target_new, device)
    plasticity, _ = hungarian_mcc(Z_target, target_eval["Z"], target_active)
    n_new = len(decision["g_new"])
    new_truth = [i for i in target_active if i >= N_PER_LAYER]
    new_mcc, _ = (hungarian_mcc(Z_target[:, :, N_PER_LAYER:], target_eval["Z"], new_truth)
                  if n_new and new_truth else (float("nan"), []))

    X_old_pca = basis_old.project(old_eval["X"])
    Z_old_ref = X_old_pca @ old_model.F_old
    old_reference, pairs = hungarian_mcc(Z_old_ref, old_eval["Z"], OLD_FACTORS)
    old_mapping = {module: truth for module, truth, _ in pairs}
    X_old_new = ((old_eval["X"] - basis_old.mean) @ V_new if n_new else
                 torch.zeros(old_eval["X"].shape[0], 2, 0))
    Z_old_all = estimate_all_modules(model, X_old_pca, X_old_new, device)
    post_mcc, _ = hungarian_mcc(Z_old_all, old_eval["Z"], OLD_FACTORS)
    selected_mcc, _ = hungarian_mcc(estimate_Z(model, X_old_pca, X_old_new, device),
                                    old_eval["Z"], OLD_FACTORS)
    stats = gate_stats(torch.tensor(decision["g_old"]), old_mapping, target_active)
    return {
        "plasticity_mcc": plasticity, "new_factor_mcc": new_mcc,
        "old_reference_MCC": old_reference,
        "post_adaptation_top_k_MCC": post_mcc,
        "forgetting": max(0.0, old_reference - post_mcc),
        "selected_branch_old_MCC_aux": selected_mcc,
        "selected_branch_forgetting_aux": max(0.0, old_reference - selected_mcc),
        "gate_acc": stats["gate_acc"], **stats,
    }


def one_run(case: str, target_active: Sequence[int], data_seed: int, train_seed: int,
            args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    data = generate_three_layer_data(args.T, args.obs_dim, args.latent_scale, data_seed)
    # Stage 1: only this block sees old observations.
    old_stage1 = data.scenarios["D1_L1"]
    old_basis = fit_pca_basis(old_stage1["X"], N_PER_LAYER)
    old_model = fit_old_model_state(old_stage1["X"], old_basis, N_PER_LAYER,
                                    random_state=70_000 + data_seed,
                                    persistence_threshold=args.persistence_threshold)

    # Stage 2: model state + target observations only.
    target_X = data.scenarios[case]["X"]
    set_seed(train_seed)
    model, decision, X_target_old, X_target_new, V_new = adapt_without_replay(
        old_model, target_X, args, device, target_ica_seed=80_000 + data_seed)

    # Offline evaluation boundary after adaptation has returned.
    metrics = evaluate_after_adaptation(model, decision, V_new, data.scenarios["D1_L1"],
                                        data.scenarios[case], old_model, target_active,
                                        X_target_old, X_target_new, device)
    return {"case": case, "data_seed": data_seed, "train_seed": train_seed,
            **metrics, **decision, **state_metadata(old_model)}


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
    p.add_argument("--out_dir", type=Path, default=HERE / "results" / "replayfree_c11_c17")
    p.add_argument("--data_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--train_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--cases", nargs="+", default=[name for name, _ in SCENARIOS])
    p.add_argument("--T", type=int, default=5_000)
    p.add_argument("--obs_dim", type=int, default=20)
    p.add_argument("--latent_scale", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--tau_start", type=float, default=2.0)
    p.add_argument("--tau_end", type=float, default=0.3)
    p.add_argument("--w_B_sp", type=float, default=0.01)
    p.add_argument("--rank_rtol", type=float, default=1e-3)
    p.add_argument("--persistence_threshold", type=float, default=0.90)
    p.add_argument("--no_target_split_consensus", action="store_true")
    p.add_argument("--freeze_new_ica_encoder", action=argparse.BooleanOptionalAction, default=True,
                   help="preserve target-only ICA source coordinates while training dynamics")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    valid = dict(SCENARIOS)
    unknown = set(args.cases) - set(valid)
    if unknown:
        p.error(f"unknown cases: {sorted(unknown)}")
    if args.obs_dim < 15:
        p.error("obs_dim must be at least 15")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args) | {"out_dir": str(args.out_dir)}, indent=2, default=str))
    path = args.out_dir / "metrics.csv"
    rows = read_csv(path) if args.resume else []
    completed = {(r["case"], int(r["data_seed"]), int(r["train_seed"])) for r in rows}
    for data_seed in args.data_seeds:
        for train_seed in args.train_seeds:
            for case in args.cases:
                key = (case, data_seed, train_seed)
                if key in completed:
                    print(f"skip completed {case} data={data_seed} train={train_seed}", flush=True)
                    continue
                row = one_run(case, valid[case], data_seed, train_seed, args, torch.device(args.device))
                rows.append(row); completed.add(key); write_csv(rows, path)
                print(f"{case} data={data_seed} train={train_seed}: gate={row['gate_acc']:.3f} "
                      f"plasticity={row['plasticity_mcc']:.3f} forgetting={row['forgetting']:.6f}", flush=True)
    write_csv(rows, path)
    print(f"completed {len(rows)} fits", flush=True)


if __name__ == "__main__":
    main()
