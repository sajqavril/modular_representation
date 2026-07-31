#!/usr/bin/env python3
"""Selected profile-MoRe adaptation on the three-layer C11--C17 benchmark."""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
from typing import Any, Dict, Iterable
import torch

from more_core import CoordBasis, TrainConfig, estimate_Z, expand_basis, fit_pca_basis, set_seed, train_more
from run_c11_c17 import (DeepMoRe, N_PER_LAYER, OLD_FACTORS, SCENARIOS, estimate_all_modules,
                         generate_three_layer_data, gate_stats, hungarian_mcc)
from more_adaptation import (ProfileOldState, fit_profile_old_state, profile_span_gate,
                             target_aligned_old_coordinates, target_profile_new_initialization)


def _basis(state): return CoordBasis(state.basis_V, state.basis_mean)
def _install(model, gate):
    model.gate_old.logits.data.copy_(torch.where(gate > 0, torch.full_like(gate, 12.), torch.full_like(gate, -12.)))
    model.gate_old.logits.requires_grad_(False)
    if model.gate_new is not None: model.gate_new.logits.data.fill_(12.); model.gate_new.logits.requires_grad_(False)


def adapt_profile_more(state: ProfileOldState, target_x, args, device, seed):
    decision = profile_span_gate(state, target_x, threshold=args.threshold, rank_rtol=args.rank_rtol)
    old_basis = _basis(state); _, v_new = expand_basis(target_x, old_basis, int(decision['n_new']))
    x_old = old_basis.project(target_x); x_new = (target_x-old_basis.mean) @ v_new if decision['n_new'] else torch.zeros(target_x.shape[0], 2, 0)
    model = DeepMoRe(N_PER_LAYER, int(decision['n_new']), N_PER_LAYER, int(decision['n_new']), state.F_old, state.B_old); _install(model, decision['g_old'])
    init = {'target_match': []}
    if decision['n_new']:
        init = target_profile_new_initialization(state, target_x, decision, x_new, lr=args.lr, epochs=args.target_id_epochs, device=device, fit_basis=fit_pca_basis, set_seed=set_seed, seed=seed+300_000, source_only=True)
        model.F_new.data.copy_(init['F_new'].to(device)); model.F_new.requires_grad_(False); x_old = target_aligned_old_coordinates(state, target_x, init)
    cfg = TrainConfig(name='profile_more_C11_C17', use_align_T=False, freeze_B_old=True, use_old_gates=True, use_new_gates=True, epochs=args.epochs, lr=args.lr, warmup_frac=0., w_sp_old=0., w_sp_new=0., w_B_sp=args.w_B_sp, w_gate_binary=0., gate_lr_factor=0.)
    if any(p.requires_grad for p in model.parameters()): set_seed(seed); train_more(model, x_old, x_new, cfg, device)
    return model, decision, x_old, x_new, v_new, init


def one(data, case, target_active, data_seed, train_seed, args, device):
    old, target = data.scenarios['D1_L1'], data.scenarios[case]
    state, z_ref = fit_profile_old_state(old['X'], N_PER_LAYER, lr=args.lr, epochs=args.old_epochs, device=device, fit_basis=fit_pca_basis, set_seed=set_seed, seed=100_000+100*data_seed+train_seed)
    model, decision, x_old, x_new, v_new, init = adapt_profile_more(state, target['X'], args, device, 200_000+100*data_seed+train_seed)
    old_ref, pairs = hungarian_mcc(z_ref, old['Z'], OLD_FACTORS); mapping = {i: truth for i, truth, _ in pairs}
    plasticity, _ = hungarian_mcc(estimate_Z(model, x_old, x_new, device), target['Z'], target_active)
    basis = _basis(state); old_x = basis.project(old['X']); old_new = (old['X']-basis.mean) @ v_new if decision['n_new'] else torch.zeros(old['X'].shape[0], 2, 0)
    post, _ = hungarian_mcc(estimate_all_modules(model, old_x, old_new, device), old['Z'], OLD_FACTORS); stats = gate_stats(decision['g_old'], mapping, target_active)
    return {'case':case, 'data_seed':data_seed, 'train_seed':train_seed, **stats, 'plasticity_mcc':plasticity, 'old_reference_MCC':old_ref, 'post_adaptation_top_k_MCC':post, 'forgetting':max(0., old_ref-post), 'g_old':decision['g_old'].tolist(), 'gate_scores':decision['scores'].tolist(), 'target_rank':decision['target_rank'], 'n_new':decision['n_new'], 'target_profile_match':init['target_match']}


def write(rows: Iterable[Dict[str, Any]], path: Path):
    rows=list(rows); fields=sorted({k for r in rows for k in r})
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in rows: w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out_dir',type=Path,default=Path(__file__).resolve().parent/'results'/'profile_more_c11_c17'); p.add_argument('--cases',nargs='+',default=[name for name,_ in SCENARIOS]); p.add_argument('--data_seeds',type=int,nargs='+',default=[0,1,2]); p.add_argument('--train_seeds',type=int,nargs='+',default=[0]); p.add_argument('--T',type=int,default=5000); p.add_argument('--obs_dim',type=int,default=20); p.add_argument('--latent_scale',type=float,default=.1); p.add_argument('--old_epochs',type=int,default=3000); p.add_argument('--target_id_epochs',type=int,default=4000); p.add_argument('--epochs',type=int,default=1000); p.add_argument('--lr',type=float,default=.003); p.add_argument('--w_B_sp',type=float,default=.01); p.add_argument('--rank_rtol',type=float,default=1e-3); p.add_argument('--threshold',type=float,default=None); p.add_argument('--device',default='cpu'); args=p.parse_args()
    unknown=set(args.cases)-{name for name,_ in SCENARIOS}
    if unknown: p.error(f'unknown cases: {sorted(unknown)}')
    args.out_dir.mkdir(parents=True,exist_ok=True); (args.out_dir/'args.json').write_text(json.dumps(vars(args)|{'out_dir':str(args.out_dir)},indent=2,default=str)); rows=[]; targets=dict(SCENARIOS)
    for ds in args.data_seeds:
        data=generate_three_layer_data(args.T,args.obs_dim,args.latent_scale,ds)
        for ts in args.train_seeds:
            for case in args.cases:
                row=one(data,case,targets[case],ds,ts,args,torch.device(args.device)); rows.append(row); write(rows,args.out_dir/'metrics.csv'); print(f"{case}: gate={row['gate_acc']:.3f} plasticity={row['plasticity_mcc']:.3f} forgetting={row['forgetting']:.6f}",flush=True)


if __name__=='__main__': main()
