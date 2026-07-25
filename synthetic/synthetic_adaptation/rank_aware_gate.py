#!/usr/bin/env python3
"""Rank-aware hard-cardinality gate for the synthetic MoRe adaptation study.

The original gate applies independent L0 penalties to all old modules.  In
C4/C5, that objective can retain both old modules even though the target has
three intrinsic factors and two expansion modules are explicitly requested.

This module estimates both quantities from unlabeled observations:

    retained_old = significant persistent old ICA directions
    added_new    = target intrinsic rank - retained_old.

No scenario name, target latent, active-factor list, or predeclared new-module
count is used.  The returned gate is fixed during adaptation: training cannot
undo a discrete structural decision merely because correlated new factors
project into the old subspace.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class GateDecision:
    """A reproducible structural gate decision."""

    g_old: torch.Tensor
    g_new: torch.Tensor
    target_rank: int
    n_retained_old: int
    score_order: torch.Tensor
    old_scores: torch.Tensor
    persistence_threshold: Optional[float] = None


@torch.no_grad()
def effective_rank(X: torch.Tensor, rtol: float = 1e-3) -> int:
    """Numerical rank of centered observations, stable across scale changes."""
    flat = X.reshape(-1, X.shape[-1]).float()
    singular_values = torch.linalg.svdvals(flat - flat.mean(dim=0, keepdim=True))
    if singular_values.numel() == 0 or singular_values[0] <= 0:
        return 0
    return int((singular_values > rtol * singular_values[0]).sum().item())


@torch.no_grad()
def rank_aware_gate(
    X_target: torch.Tensor,
    a_old: torch.Tensor,
    n_new_candidates: int,
    rank_rtol: float = 1e-3,
    selection_scores: Optional[torch.Tensor] = None,
) -> GateDecision:
    """Choose old/new gates from reconstruction capacity and temporal evidence.

    ``n_new_candidates`` is the number of modules added for the next layer.
    All of them remain enabled so the adaptation can generate the requested
    new factors.  The old gate count is the target rank left after reserving
    this expansion capacity; ties are deterministically broken by index.
    """
    a_old = a_old.detach().float().cpu()
    scores = a_old if selection_scores is None else selection_scores.detach().float().cpu()
    if scores.shape != a_old.shape:
        raise ValueError(f"selection scores shape {tuple(scores.shape)} != {tuple(a_old.shape)}")
    m_old = int(scores.numel())
    rank = effective_rank(X_target.detach().cpu(), rank_rtol)
    n_old = max(0, min(m_old, rank - int(n_new_candidates)))

    # stable=True makes a score tie deterministic and independent of hardware.
    order = torch.argsort(scores, descending=True, stable=True)
    g_old = torch.zeros(m_old, dtype=torch.float32)
    g_old[order[:n_old]] = 1.0
    g_new = torch.ones(int(n_new_candidates), dtype=torch.float32)
    return GateDecision(g_old, g_new, rank, n_old, order, scores)


def _normalized_columns(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=0, keepdims=True).clip(min=1e-12)


def _best_assignment(similarity: np.ndarray) -> tuple[int, ...]:
    n_rows, n_columns = similarity.shape
    if n_rows != n_columns:
        raise ValueError("ICA assignment currently requires equal old component counts")
    return max(permutations(range(n_columns)),
               key=lambda perm: sum(similarity[i, perm[i]] for i in range(n_rows)))


def _fit_ica(samples: np.ndarray, n_components: int, random_state: int):
    from sklearn.decomposition import FastICA
    return FastICA(n_components=n_components, whiten="unit-variance", random_state=random_state,
                   max_iter=2000, tol=1e-5).fit(samples)


@torch.no_grad()
def anchored_ica_scores(
    X_old: torch.Tensor,
    X_target: torch.Tensor,
    Z_old: torch.Tensor,
    n_target_components: Optional[int] = None,
    random_state: int = 0,
) -> torch.Tensor:
    """Score frozen old modules by persistence of their ICA mixing direction.

    At t0, the synthetic factors are independent.  FastICA therefore recovers
    the old mixing directions up to permutation/sign.  We recover target t0
    directions independently and use the largest absolute cosine match as a
    persistence score.  Finally, a maximum-correlation assignment maps ICA
    components back to the *existing frozen encoder columns*, making the gate
    explicitly permutation-aware without using latent labels.
    """
    try:
        from sklearn.decomposition import FastICA
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("anchored_ica gate requires scikit-learn") from error

    old = X_old[:, 0, :].detach().cpu().numpy()
    target = X_target[:, 0, :].detach().cpu().numpy()
    z_old = Z_old[:, 0, :].detach().cpu().numpy()
    m_old = z_old.shape[1]
    n_target = n_target_components or effective_rank(X_target)
    if n_target < 1:
        return torch.zeros(m_old, dtype=torch.float32)

    ica_old = _fit_ica(old, m_old, random_state)
    ica_target = _fit_ica(target, n_target, random_state)
    old_mixing, target_mixing = ica_old.mixing_, ica_target.mixing_
    old_mixing = _normalized_columns(old_mixing)
    target_mixing = _normalized_columns(target_mixing)
    ica_persistence = np.abs(old_mixing.T @ target_mixing).max(axis=1)

    old_sources = ica_old.transform(old)
    z_old = (z_old - z_old.mean(axis=0)) / np.maximum(z_old.std(axis=0), 1e-12)
    old_sources = (old_sources - old_sources.mean(axis=0)) / np.maximum(old_sources.std(axis=0), 1e-12)
    correspondence = np.abs(z_old.T @ old_sources / max(z_old.shape[0] - 1, 1))
    assignment = _best_assignment(correspondence)
    module_scores = np.array([ica_persistence[assignment[i]] for i in range(m_old)], dtype=np.float32)
    return torch.from_numpy(module_scores)


@torch.no_grad()
def leakage_free_ica_gate(
    X_old: torch.Tensor,
    X_target: torch.Tensor,
    Z_old: torch.Tensor,
    rank_rtol: float = 1e-3,
    n_bootstrap: int = 4,
    stability_margin: float = 0.02,
    random_state: int = 0,
    target_split_consensus: bool = True,
    threshold_override: Optional[float] = None,
) -> GateDecision:
    """Infer retained and added dimensions without a declared expansion size.

    A persistent old direction must match target ICA directions at least as
    well as an independently re-estimated version of itself on old-data splits.
    This calibrated threshold avoids latent-label or scenario-count leakage.
    Target split consensus prevents a chance target-ICA match from being kept.
    ``threshold_override`` supports a label-free fixed-threshold ablation.
    """
    try:
        from sklearn.decomposition import FastICA  # noqa: F401
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("leakage_free_ica gate requires scikit-learn") from error
    old = X_old[:, 0, :].detach().cpu().numpy()
    target = X_target[:, 0, :].detach().cpu().numpy()
    z_old = Z_old[:, 0, :].detach().cpu().numpy()
    m_old = z_old.shape[1]
    target_rank = effective_rank(X_target, rank_rtol)
    if target_rank == 0:
        empty = torch.zeros(m_old, dtype=torch.float32)
        return GateDecision(empty, torch.zeros(0), 0, 0, torch.arange(m_old), empty, 1.0)

    old_ica = _fit_ica(old, m_old, random_state)
    old_mix = _normalized_columns(old_ica.mixing_)
    old_sources = old_ica.transform(old)
    z_standard = (z_old - z_old.mean(axis=0)) / np.maximum(z_old.std(axis=0), 1e-12)
    s_standard = (old_sources - old_sources.mean(axis=0)) / np.maximum(old_sources.std(axis=0), 1e-12)
    module_to_ica = _best_assignment(np.abs(z_standard.T @ s_standard / max(len(old)-1, 1)))

    target_ica = _fit_ica(target, target_rank, random_state)
    target_mix = _normalized_columns(target_ica.mixing_)
    full_ica_scores = np.abs(old_mix.T @ target_mix).max(axis=1)

    rng = np.random.default_rng(random_state)
    old_self_scores, target_split_scores = [], []
    for replicate in range(n_bootstrap):
        old_indices = rng.choice(len(old), size=len(old) // 2, replace=False)
        target_indices = rng.choice(len(target), size=len(target) // 2, replace=False)
        old_split_mix = _normalized_columns(_fit_ica(old[old_indices], m_old, random_state + replicate + 1).mixing_)
        target_split_mix = _normalized_columns(_fit_ica(target[target_indices], target_rank, random_state + replicate + 101).mixing_)
        old_self_scores.extend(np.abs(old_mix.T @ old_split_mix).max(axis=1).tolist())
        target_split_scores.append(np.abs(old_mix.T @ target_split_mix).max(axis=1))

    # Lower tail of self-recovery, relaxed slightly for finite-sample ICA noise.
    threshold = float(np.quantile(old_self_scores, 0.05) - stability_margin)
    if threshold_override is not None:
        threshold = float(threshold_override)
    if target_split_consensus:
        consensus_ica_scores = np.minimum(full_ica_scores, np.min(np.stack(target_split_scores), axis=0))
    else:
        consensus_ica_scores = full_ica_scores
    module_scores = torch.tensor([consensus_ica_scores[module_to_ica[i]] for i in range(m_old)], dtype=torch.float32)
    order = torch.argsort(module_scores, descending=True, stable=True)
    keep = module_scores >= threshold
    # A target of rank r cannot contain more than r retained independent directions.
    if int(keep.sum()) > target_rank:
        keep = torch.zeros_like(keep)
        keep[order[:target_rank]] = True
    n_retained = int(keep.sum().item())
    n_new = max(0, target_rank - n_retained)
    return GateDecision(keep.float(), torch.ones(n_new), target_rank, n_retained,
                        order, module_scores, threshold)


def install_fixed_gate(model, decision: GateDecision, logit: float = 12.0) -> None:
    """Install a hard structural decision into a MoRe instance.

    We use saturated logits and freeze them, rather than multiplying the
    features by a score.  This retains a selected old module at full strength
    and lets selected z3/z4 modules receive gradients from the first step.
    """
    old_logits = torch.where(
        decision.g_old > 0,
        torch.full_like(decision.g_old, logit),
        torch.full_like(decision.g_old, -logit),
    )
    model.gate_old.logits.data.copy_(old_logits.to(model.gate_old.logits.device))
    model.gate_old.logits.requires_grad_(False)
    if model.gate_new is not None:
        new_logits = torch.full_like(model.gate_new.logits, logit)
        model.gate_new.logits.data.copy_(new_logits.to(model.gate_new.logits.device))
        model.gate_new.logits.requires_grad_(False)


def decision_metadata(decision: GateDecision) -> Dict:
    return {
        "target_rank": decision.target_rank,
        "n_retained_old": decision.n_retained_old,
        "score_order": decision.score_order.tolist(),
        "old_selection_scores": decision.old_scores.tolist(),
        "persistence_threshold": decision.persistence_threshold,
        "g_old": decision.g_old.tolist(),
        "g_new": decision.g_new.tolist(),
    }
