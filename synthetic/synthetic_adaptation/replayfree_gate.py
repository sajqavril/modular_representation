#!/usr/bin/env python3
"""Replay-free structural gate for sequential synthetic adaptation.

The stage-2 entry point in this module deliberately accepts only an
``OldModelState`` and target observations.  It has no old-data argument and
does not require old latent outputs, active-factor annotations, or a declared
expansion size.

At the end of stage 1, an identifiable ICA encoder is fitted on the old
PCA-coordinates and stored as the model's frozen encoder.  The physical source
directions are recoverable from the stored PCA basis and encoder inverse:

    anchor_j = normalize(V_old @ inverse(F_old).T[:, j]).

Therefore the old samples can be discarded.  At stage 2, target ICA directions
are compared against those stored model directions; no replay is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from rank_aware_gate import GateDecision, effective_rank


@dataclass(frozen=True)
class ReplayFreeOldModelState:
    """All stage-1 state available at adaptation; contains no old samples."""

    basis_V: torch.Tensor
    basis_mean: torch.Tensor
    F_old: torch.Tensor
    B_old: torch.Tensor
    persistence_threshold: float = 0.90

    @property
    def n_old(self) -> int:
        return int(self.F_old.shape[1])

    @torch.no_grad()
    def source_anchors(self) -> torch.Tensor:
        """Normalized raw-observation source directions derived from model weights."""
        # X_pca @ F_old = Z, hence X_pca = Z @ inverse(F_old).  The
        # observation-space direction of source j is row j of that inverse,
        # represented as a column: inverse(F_old).T[:, j].
        mixing_coordinates = torch.linalg.inv(self.F_old.float()).T
        raw_directions = self.basis_V.float() @ mixing_coordinates
        return raw_directions / raw_directions.norm(dim=0, keepdim=True).clamp_min(1e-12)


def _fit_ica(samples: np.ndarray, n_components: int, random_state: int):
    from sklearn.decomposition import FastICA
    return FastICA(n_components=n_components, whiten="unit-variance",
                   random_state=random_state, max_iter=2_000, tol=1e-5).fit(samples)


@torch.no_grad()
def fit_old_model_state(old_X: torch.Tensor, basis, n_old: int,
                        random_state: int, persistence_threshold: float = 0.90,
                        encoder_column_norm: float = 0.3) -> ReplayFreeOldModelState:
    """Stage 1 only: learn canonical old encoder/dynamics before data disposal.

    This is the sole function here that accepts old observations.  It must run
    before old data are discarded; its returned state contains only model
    weights and the required preprocessing transformation.
    """
    X_old_pca = basis.project(old_X).detach().cpu()
    samples = X_old_pca[:, 0, :].numpy()
    ica = _fit_ica(samples, n_old, random_state)
    F_old = torch.from_numpy(ica.components_.T).float()
    # ICA has arbitrary source scales. Keep directions while matching the
    # numerical scale used by MoRe's learned encoder initialization; otherwise
    # the log-determinant optimizer can start from excessively large columns.
    F_old = F_old / F_old.norm(dim=0, keepdim=True).clamp_min(1e-12) * encoder_column_norm
    Z_old = X_old_pca @ F_old
    # Least-squares dynamics in the same canonical source coordinates.
    B_old = torch.linalg.lstsq(Z_old[:, 0, :], Z_old[:, 1, :]).solution.T.contiguous()
    return ReplayFreeOldModelState(
        basis_V=basis.V.detach().cpu().clone(),
        basis_mean=basis.mean.detach().cpu().clone(),
        F_old=F_old, B_old=B_old,
        persistence_threshold=float(persistence_threshold),
    )


@torch.no_grad()
def replayfree_target_gate(old_model: ReplayFreeOldModelState, X_target: torch.Tensor,
                           rank_rtol: float = 1e-3,
                           target_ica_seed: int = 0,
                           target_split_consensus: bool = True) -> GateDecision:
    """Stage 2 decision using only the saved model and new observations.

    No old data, old latent outputs, scenario name, target latent annotations,
    old active set, or expansion count are accepted by this function.
    """
    target_rank = effective_rank(X_target, rank_rtol)
    m_old = old_model.n_old
    if target_rank == 0:
        empty = torch.zeros(m_old, dtype=torch.float32)
        return GateDecision(empty, torch.zeros(0), 0, 0, torch.arange(m_old), empty,
                            old_model.persistence_threshold)

    target_samples = X_target[:, 0, :].detach().cpu().numpy()
    target_ica = _fit_ica(target_samples, target_rank, target_ica_seed)
    target_mixing = torch.from_numpy(target_ica.mixing_).float()
    target_mixing = target_mixing / target_mixing.norm(dim=0, keepdim=True).clamp_min(1e-12)
    anchors = old_model.source_anchors()
    full_scores = (anchors.T @ target_mixing).abs().max(dim=1).values

    # Optional target-only stability check. It uses independent subsets of new
    # data and therefore remains replay-free.
    if target_split_consensus and len(target_samples) >= 20:
        rng = np.random.default_rng(target_ica_seed)
        split_scores = []
        for replicate in range(2):
            idx = rng.choice(len(target_samples), size=len(target_samples) // 2, replace=False)
            split = _fit_ica(target_samples[idx], target_rank, target_ica_seed + 101 + replicate)
            mix = torch.from_numpy(split.mixing_).float()
            mix = mix / mix.norm(dim=0, keepdim=True).clamp_min(1e-12)
            split_scores.append((anchors.T @ mix).abs().max(dim=1).values)
        scores = torch.minimum(full_scores, torch.stack(split_scores).min(dim=0).values)
    else:
        scores = full_scores

    order = torch.argsort(scores, descending=True, stable=True)
    keep = scores >= old_model.persistence_threshold
    # Never retain more independent directions than target rank.
    if int(keep.sum()) > target_rank:
        keep = torch.zeros_like(keep)
        keep[order[:target_rank]] = True
    n_retained = int(keep.sum().item())
    n_new = max(0, target_rank - n_retained)
    return GateDecision(keep.float(), torch.ones(n_new), target_rank, n_retained,
                        order, scores, old_model.persistence_threshold)


@torch.no_grad()
def initialize_new_encoder_target_ica(F_new: torch.Tensor, X_target_new: torch.Tensor,
                                      random_state: int, encoder_column_norm: float = 0.3) -> None:
    """Target-only ICA initialization for newly allocated residual modules."""
    if F_new.numel() == 0:
        return
    samples = X_target_new[:, 0, :].detach().cpu().numpy()
    ica = _fit_ica(samples, F_new.shape[1], random_state)
    directions = torch.from_numpy(ica.components_.T).to(F_new)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    F_new.copy_(directions * encoder_column_norm)


def state_metadata(state: ReplayFreeOldModelState) -> Dict:
    """Auditable description of stored model tensors; deliberately no samples."""
    return {
        "state_tensors": {
            "basis_V": list(state.basis_V.shape), "basis_mean": list(state.basis_mean.shape),
            "F_old": list(state.F_old.shape), "B_old": list(state.B_old.shape),
        },
        "persistence_threshold": state.persistence_threshold,
        "contains_old_samples": False,
        "contains_old_labels_or_active_metadata": False,
    }
