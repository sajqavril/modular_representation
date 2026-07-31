#!/usr/bin/env python3
"""Selected profile-MoRe adaptation on C1--C7, without old-data replay."""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
from typing import Any, Dict, Iterable
import torch

from more_core import (ALL_TRANSITIONS, CoordBasis, TrainConfig, best_mcc, estimate_Z,
                       expand_basis, fit_pca_basis, gate_accuracy, generate_data, set_seed, train_more)
from run_c11_c17 import DeepMoRe
from more_adaptation import (ProfileOldState, fit_profile_old_state, profile_span_gate,
                             target_aligned_old_coordinates, target_profile_new_initialization)


def _basis(state: ProfileOldState) -> CoordBasis:
    return CoordBasis(state.basis_V, state.basis_mean)


def _install_gate(model: DeepMoRe, g_old: torch.Tensor) -> None:
    model.gate_old.logits.data.copy_(torch.where(g_old > 0, torch.full_like(g_old, 12.), torch.full_like(g_old, -12.)))
    model.gate_old.logits.requires_grad_(False)
    if model.gate_new is not None:
        model.gate_new.logits.data.fill_(12.); model.gate_new.logits.requires_grad_(False)


def adapt_profile_more(state: ProfileOldState, target_x: torch.Tensor, args, device, seed: int):
    """Deployment API: saved state + target observations only."""
    decision = profile_span_gate(state, target_x, threshold=args.threshold, rank_rtol=args.rank_rtol)
    old_basis = _basis(state)
    _, v_new = expand_basis(target_x, old_basis, int(decision['n_new']))
    x_old = old_basis.project(target_x)
    x_new = ((target_x - old_basis.mean) @ v_new if decision['n_new'] else
             torch.zeros(target_x.shape[0], 2, 0))
    model = DeepMoRe(state.F_old.shape[0], int(decision['n_new']), state.F_old.shape[1],
                     int(decision['n_new']), state.F_old, state.B_old)
    _install_gate(model, decision['g_old'])
    init = {'target_match': [], 'new_target_indices': []}
    if decision['n_new']:
        init = target_profile_new_initialization(
            state, target_x, decision, x_new, lr=args.lr, epochs=args.target_id_epochs,
            device=device, fit_basis=fit_pca_basis, set_seed=set_seed, seed=seed + 300_000,
            source_only=True)
        model.F_new.data.copy_(init['F_new'].to(device)); model.F_new.requires_grad_(False)
        x_old = target_aligned_old_coordinates(state, target_x, init)
    cfg = TrainConfig(name='profile_more_fixed_gate', use_align_T=False, freeze_B_old=True,
                      use_old_gates=True, use_new_gates=True, epochs=args.epochs, lr=args.lr,
                      warmup_frac=0., w_sp_old=0., w_sp_new=0., w_B_sp=args.w_B_sp,
                      w_gate_binary=0., gate_lr_factor=0.)
    if any(p.requires_grad for p in model.parameters()):
        set_seed(seed); train_more(model, x_old, x_new, cfg, device)
    return model, decision, x_old, x_new, v_new, init


@torch.no_grad()
def _all_modules(model, x_old, x_new, device):
    old = model.gate_old.logits.detach().clone()
    new = model.gate_new.logits.detach().clone() if model.gate_new is not None else None
    try:
        model.gate_old.logits.data.fill_(12.)
        if model.gate_new is not None: model.gate_new.logits.data.fill_(12.)
        return estimate_Z(model, x_old, x_new, device)
    finally:
        model.gate_old.logits.data.copy_(old)
        if model.gate_new is not None: model.gate_new.logits.data.copy_(new)


def one(case: int, data_seed: int, train_seed: int, args, device) -> Dict[str, Any]:
    name, old_name, target_name, old_active, target_active, old_dim, *_ = ALL_TRANSITIONS[case]
    data = generate_data(args.T, args.obs_dim, args.noise_scale, data_seed)
    old, target = data.scenarios[old_name], data.scenarios[target_name]
    state, z_ref = fit_profile_old_state(old['X'], old_dim, lr=args.lr, epochs=args.old_epochs,
        device=device, fit_basis=fit_pca_basis, set_seed=set_seed,
        seed=100_000 + 100 * data_seed + train_seed)
    model, decision, x_old, x_new, v_new, init = adapt_profile_more(
        state, target['X'], args, device, 200_000 + 100 * data_seed + train_seed)
    # Evaluation only: old data and labels are not available to adaptation.
    old_ref, pairs = best_mcc(z_ref, old['Z'], old_active); old_map = {i: truth for i, truth, _ in pairs}
    plasticity, _ = best_mcc(estimate_Z(model, x_old, x_new, device), target['Z'], target_active)
    old_basis = _basis(state); x_old_old = old_basis.project(old['X'])
    x_old_new = (old['X'] - old_basis.mean) @ v_new if decision['n_new'] else torch.zeros(old['X'].shape[0], 2, 0)
    post, _ = best_mcc(_all_modules(model, x_old_old, x_old_new, device), old['Z'], old_active)
    stats = gate_accuracy(decision['g_old'], old_map, target_active)
    return {'case': name, 'data_seed': data_seed, 'train_seed': train_seed, **stats,
            'plasticity_mcc': plasticity, 'old_reference_MCC': old_ref,
            'post_adaptation_top_k_MCC': post, 'forgetting': max(0., old_ref - post),
            'g_old': decision['g_old'].tolist(), 'gate_scores': decision['scores'].tolist(),
            'target_rank': decision['target_rank'], 'n_new': decision['n_new'],
            'target_profile_match': init['target_match']}


def _write(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    rows = list(rows); fields = sorted({key for row in rows for key in row})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument('--out_dir', type=Path, default=Path(__file__).resolve().parent/'results'/'profile_more_c1_c7')
    p.add_argument('--cases', type=int, nargs='+', default=list(range(7))); p.add_argument('--data_seeds', type=int, nargs='+', default=[0,1,2]); p.add_argument('--train_seeds', type=int, nargs='+', default=[0])
    p.add_argument('--T', type=int, default=5000); p.add_argument('--obs_dim', type=int, default=8); p.add_argument('--noise_scale', type=float, default=.1)
    p.add_argument('--old_epochs', type=int, default=2000); p.add_argument('--target_id_epochs', type=int, default=2000); p.add_argument('--epochs', type=int, default=1500); p.add_argument('--lr', type=float, default=.005); p.add_argument('--w_B_sp', type=float, default=.01); p.add_argument('--rank_rtol', type=float, default=1e-3); p.add_argument('--threshold', type=float, default=None); p.add_argument('--device', default='cpu')
    args = p.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True); (args.out_dir/'args.json').write_text(json.dumps(vars(args)|{'out_dir':str(args.out_dir)}, indent=2, default=str))
    rows = []
    for data_seed in args.data_seeds:
        for train_seed in args.train_seeds:
            for case in args.cases:
                row = one(case, data_seed, train_seed, args, torch.device(args.device)); rows.append(row); _write(rows, args.out_dir/'metrics.csv')
                print(f"{row['case']}: gate={row['gate_acc']:.3f} plasticity={row['plasticity_mcc']:.3f} forgetting={row['forgetting']:.6f}", flush=True)


if __name__ == '__main__': main()
