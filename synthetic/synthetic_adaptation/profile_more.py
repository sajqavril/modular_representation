"""Profile-aligned, replay-free MoRe identification and adaptation utilities.

The deployment boundary is explicit: a ``ProfileOldState`` contains only
saved model parameters and decoder directions.  Every stage-2 function takes
that state and target observations; it never accepts old examples, labels,
or a prescribed expansion width.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class ProfileIdentifier(nn.Module):
    """Linear MoRe identifier with a scale-profiled Laplace likelihood."""

    def __init__(self, coord_dim: int, n_modules: int):
        super().__init__()
        self.F = nn.Parameter(.05 * torch.randn(coord_dim, n_modules))
        self.B = nn.Parameter(.8 * torch.eye(n_modules))

    def forward(self, x: torch.Tensor, *, source_only: bool = False) -> Dict[str, torch.Tensor]:
        z = x @ self.F
        innovation = z[:, 1] - z[:, 0] @ self.B.T
        gram = self.F.T @ self.F + 1e-6 * torch.eye(self.F.shape[1], device=x.device)
        logdet = torch.linalg.slogdet(gram)[1]
        source = z[:, 0].abs().mean(0).clamp_min(1e-8).log().mean()
        if source_only:
            likelihood = source - .5 * logdet / self.F.shape[1]
        else:
            residual = innovation.abs().mean(0).clamp_min(1e-8).log().mean()
            likelihood = source + residual - logdet / self.F.shape[1]
        return {'loss': likelihood, 'F': self.F, 'B': self.B, 'Z': z, 'E': innovation}


def _decoder_directions(basis_v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    decoded = basis_v @ torch.linalg.pinv(f).T
    return decoded / decoded.norm(dim=0, keepdim=True).clamp_min(1e-8)


def _fit_identifier(x: torch.Tensor, n_modules: int, *, lr: float, epochs: int,
                    device: torch.device, set_seed, seed: int,
                    source_only: bool) -> Dict[str, torch.Tensor]:
    set_seed(seed)
    model = ProfileIdentifier(x.shape[-1], n_modules).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xd = x.to(device)
    for _ in range(epochs):
        out = model(xd, source_only=source_only)
        opt.zero_grad(set_to_none=True); out['loss'].backward(); opt.step()
    with torch.no_grad():
        out = model(xd, source_only=source_only)
    return {key: value.detach().cpu() for key, value in out.items() if key != 'loss'}


@dataclass(frozen=True)
class ProfileOldState:
    basis_V: torch.Tensor
    basis_mean: torch.Tensor
    F_old: torch.Tensor
    B_old: torch.Tensor
    decoder_directions: torch.Tensor


def fit_profile_old_state(old_x: torch.Tensor, old_dim: int, *, lr: float, epochs: int,
                          device: torch.device, fit_basis, set_seed, seed: int,
                          source_only: bool = False) -> tuple[ProfileOldState, torch.Tensor]:
    """Fit stage 1 and return model state only; callers discard ``old_x``."""
    basis = fit_basis(old_x, old_dim)
    info = _fit_identifier(basis.project(old_x), old_dim, lr=lr, epochs=epochs,
                           device=device, set_seed=set_seed, seed=seed,
                           source_only=source_only)
    state = ProfileOldState(basis.V.detach().cpu(), basis.mean.detach().cpu(),
                            info['F'], info['B'], _decoder_directions(basis.V, info['F']))
    return state, info['Z']


@torch.no_grad()
def profile_span_gate(old_state: ProfileOldState, target_x: torch.Tensor, *,
                      threshold: float | None = None, rank_rtol: float = 1e-3) -> Dict:
    """Retain an old decoder iff it belongs to the target observation span."""
    if threshold is None:
        threshold = 1. - 10. * rank_rtol
    flat = target_x.reshape(-1, target_x.shape[-1]).float()
    _, singular, vh = torch.linalg.svd(flat - flat.mean(0, keepdim=True), full_matrices=False)
    rank = int((singular > rank_rtol * singular[0]).sum()) if len(singular) else 0
    if rank == 0:
        zero = torch.zeros(old_state.F_old.shape[1])
        return {'g_old': zero, 'scores': zero, 'target_rank': 0,
                'n_retained_old': 0, 'n_new': 0, 'threshold': threshold}
    target_basis = vh[:rank].T
    scores = (target_basis.T @ old_state.decoder_directions.float()).norm(dim=0).clamp(0, 1)
    g_old = (scores >= threshold).float()
    if int(g_old.sum()) > rank:
        g_old.zero_(); g_old[torch.argsort(scores, descending=True)[:rank]] = 1.
    retained = int(g_old.sum())
    return {'g_old': g_old.cpu(), 'scores': scores.cpu(), 'target_rank': rank,
            'n_retained_old': retained, 'n_new': rank - retained,
            'threshold': threshold}


def target_profile_new_initialization(old_state: ProfileOldState, target_x: torch.Tensor,
                                      decision: Dict, x_new: torch.Tensor, *, lr: float,
                                      epochs: int, device: torch.device, fit_basis, set_seed,
                                      seed: int, source_only: bool = True) -> Dict:
    """Target-only profile fit, one-to-one retained matching, and new-F init."""
    n_new, rank = int(decision['n_new']), int(decision['target_rank'])
    if not n_new:
        return {'F_new': torch.zeros(x_new.shape[-1], 0), 'target_match': [],
                'new_target_indices': [], 'target_basis_V': None,
                'target_basis_mean': None, 'target_F': None}
    target_basis = fit_basis(target_x, rank)
    info = _fit_identifier(target_basis.project(target_x), rank, lr=lr, epochs=epochs,
                           device=device, set_seed=set_seed, seed=seed,
                           source_only=source_only)
    target_dirs = _decoder_directions(target_basis.V, info['F'])
    retained = torch.nonzero(decision['g_old'] > .5, as_tuple=False).flatten().tolist()
    used, matches = set(), []
    if retained:
        sim = (old_state.decoder_directions[:, retained].T @ target_dirs).abs().numpy()
        rows, cols = linear_sum_assignment(-sim)
        for row, col in zip(rows.tolist(), cols.tolist()):
            matches.append((int(retained[row]), int(col), float(sim[row, col])))
            used.add(int(col))
    new_indices = [index for index in range(rank) if index not in used]
    if len(new_indices) != n_new:
        raise RuntimeError('target matching disagrees with rank-derived expansion')
    f_new = torch.linalg.lstsq(x_new[:, 0].float(), info['Z'][:, 0, new_indices].float()).solution
    return {'F_new': f_new.detach().cpu(), 'target_match': matches,
            'new_target_indices': new_indices, 'target_basis_V': target_basis.V.detach().cpu(),
            'target_basis_mean': target_basis.mean.detach().cpu(), 'target_F': info['F']}


@torch.no_grad()
def target_aligned_old_coordinates(old_state: ProfileOldState, target_x: torch.Tensor,
                                   target_init: Dict) -> torch.Tensor:
    """Transport accepted target components to the frozen old-model interface."""
    if target_init['target_F'] is None:
        raise ValueError('target profile fit is required for alignment')
    target_z = ((target_x - target_init['target_basis_mean']) @ target_init['target_basis_V']
                @ target_init['target_F'])
    desired = torch.zeros(*target_z.shape[:2], old_state.F_old.shape[1], dtype=target_z.dtype)
    for old_index, target_index, _ in target_init['target_match']:
        desired[..., old_index] = target_z[..., target_index]
    return desired @ torch.linalg.pinv(old_state.F_old)
