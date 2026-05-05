#!/usr/bin/env python3
"""
MoRe selective adaptation experiment script.

This file is intentionally self contained. It reproduces the linear sequential
latent data generation style from adaptation.ipynb and evaluates MoRe on:
  1) A: (z1,z2)       -> B: (z1,z3,z4)
  2) B: (z1,z3,z4)   -> C: (z1,z3)

It compares baselines/ablations, logs stability/plasticity MCC, gate assignment
correctness, and efficiency metrics, and saves all proposed-MoRe parts.

Example:
  python more_selective_adaptation_experiment.py --out_dir results_more --T 50000 --epochs 20000 --seeds 0 1 2

Notes:
  - The model uses an expanding observation coordinate system. Old data define an
    initial PCA basis; new residual directions are added as candidates.
  - PCA/residual directions are initialization/capacity hints only. Gates decide
    which modules are actually active.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from itertools import permutations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_cpu_np(x: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if x is None:
        return None
    return x.detach().cpu().numpy()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = 0
    trainable = 0
    by_name = {}
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            by_name[name] = n
    return {"total_params": total, "trainable_params": trainable, **{f"trainable_{k}": v for k, v in by_name.items()}}


# -----------------------------------------------------------------------------
# PCA / expanding observation coordinate system
# -----------------------------------------------------------------------------


def flatten_X(X: torch.Tensor) -> torch.Tensor:
    return X.reshape(-1, X.shape[-1])


@dataclass
class Basis:
    mean: torch.Tensor       # 1 x obs_dim
    components: torch.Tensor # obs_dim x k

    def to(self, device: torch.device | str) -> "Basis":
        return Basis(self.mean.to(device), self.components.to(device))

    @property
    def dim(self) -> int:
        return int(self.components.shape[1])


def fit_pca_basis(X: torch.Tensor, k: int) -> Basis:
    X_flat = flatten_X(X)
    mean = X_flat.mean(dim=0, keepdim=True)
    Xc = X_flat - mean
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    comps = Vh[:k].T.contiguous()
    return Basis(mean=mean, components=comps)


def transform_basis(X: torch.Tensor, basis: Basis) -> torch.Tensor:
    X_flat = flatten_X(X)
    Xc = X_flat - basis.mean.to(X.device)
    Xred = Xc @ basis.components.to(X.device)
    return Xred.reshape(*X.shape[:-1], basis.dim)


def reconstruct_from_basis(X_reduced: torch.Tensor, basis: Basis) -> torch.Tensor:
    Xr = flatten_X(X_reduced)
    X_full = Xr @ basis.components.to(X_reduced.device).T + basis.mean.to(X_reduced.device)
    return X_full.reshape(*X_reduced.shape[:-1], basis.components.shape[0])


def orthonormalize_against(V_old: torch.Tensor, V_new: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Remove old span from V_new and orthonormalize remaining columns."""
    if V_new.numel() == 0:
        return V_new
    V = V_new - V_old @ (V_old.T @ V_new)
    # QR can return fewer meaningful directions if residual rank is low. Keep columns
    # but avoid NaNs.
    Q, R = torch.linalg.qr(V, mode="reduced")
    diag = torch.abs(torch.diag(R))
    keep = diag > eps
    if keep.sum() == 0:
        return torch.zeros(V_old.shape[0], 0, device=V_old.device, dtype=V_old.dtype)
    return Q[:, keep]


def expand_basis_with_residual(X_new: torch.Tensor, old_basis: Basis, k_extra: int) -> Tuple[Basis, torch.Tensor]:
    """Append up to k_extra PCA directions of the residual after old-basis projection."""
    if k_extra <= 0:
        return old_basis, torch.empty(old_basis.components.shape[0], 0, dtype=old_basis.components.dtype)
    X_old_coords = transform_basis(X_new, old_basis)
    X_recon = reconstruct_from_basis(X_old_coords, old_basis)
    R = X_new - X_recon
    R_flat = flatten_X(R)
    R_mean = R_flat.mean(dim=0, keepdim=True)
    Rc = R_flat - R_mean
    try:
        _, _, Vh = torch.linalg.svd(Rc, full_matrices=False)
    except RuntimeError:
        # Fallback for occasional GPU SVD instability.
        _, _, Vh = torch.linalg.svd(Rc.cpu(), full_matrices=False)
        Vh = Vh.to(Rc.device)
    V_candidate = Vh[:k_extra].T.contiguous()
    V_new = orthonormalize_against(old_basis.components.to(V_candidate.device), V_candidate)
    if V_new.shape[1] < k_extra:
        # Pad with random orthogonal directions if residual rank is too low. They should
        # be gated off by a well-behaved method.
        obs_dim = old_basis.components.shape[0]
        need = k_extra - V_new.shape[1]
        rand = torch.randn(obs_dim, need, device=V_new.device)
        V_pad = orthonormalize_against(torch.cat([old_basis.components.to(V_new.device), V_new], dim=1), rand)
        if V_pad.shape[1] > need:
            V_pad = V_pad[:, :need]
        V_new = torch.cat([V_new, V_pad], dim=1)
    V_new = V_new[:, :k_extra]
    expanded = Basis(
        mean=old_basis.mean.clone(),
        components=torch.cat([old_basis.components.to(V_new.device), V_new], dim=1).contiguous(),
    )
    return expanded, V_new


# -----------------------------------------------------------------------------
# Synthetic data generation, based on adaptation.ipynb
# -----------------------------------------------------------------------------


@dataclass
class SyntheticData:
    Z_full: torch.Tensor   # T x 2 x 4
    X_full: torch.Tensor   # T x 2 x obs_dim
    mixing_A: torch.Tensor # 4 x obs_dim
    scenarios: Dict[str, Dict[str, torch.Tensor]]


def generate_base_data(T: int, noise_scale: float, obs_dim: int, seed: int) -> SyntheticData:
    set_seed(seed)
    L = 1
    lap = torch.distributions.Laplace(0.0, noise_scale)
    E1 = lap.sample((T, L + 1))
    E2 = lap.sample((T, L + 1))
    E3 = lap.sample((T, L + 1))
    E4 = lap.sample((T, L + 1))

    # first layer Z1, Z2
    Z1 = E1.clone()
    Z1[:, 1] = 0.8 * E1[:, 0] + E1[:, 1] + 0.5 * E2[:, 0]
    Z2 = E2.clone()
    Z2[:, 1] = 0.8 * E2[:, 0] + E2[:, 1] + 0.5 * E1[:, 0]

    # second layer Z3, Z4, depending on Z1/Z2 and each other.
    Z3 = E3.clone()
    Z3[:, 1] = 0.8 * E3[:, 0] + 0.6 * Z1[:, 0] + 0.8 * Z2[:, 0] + E3[:, 1] + 0.5 * E4[:, 0]
    Z4 = E4.clone()
    Z4[:, 1] = 0.8 * E4[:, 0] + 0.4 * Z1[:, 0] + 0.6 * Z2[:, 0] + E4[:, 1] + 0.5 * E3[:, 0]

    Z = torch.stack([Z1, Z2, Z3, Z4], dim=-1)
    mixing_A = torch.rand(size=(4, obs_dim))
    X = Z @ mixing_A

    scenarios = {}
    # A: z1,z2 only
    Z_A = copy.deepcopy(Z)
    Z_A[:, :, 2:] = 0.0
    X_A = Z_A @ mixing_A
    scenarios["A_z1z2"] = {"Z": Z_A, "X": X_A, "active": torch.tensor([0, 1])}

    # B: z1,z3,z4 only
    Z_B = copy.deepcopy(Z)
    Z_B[:, :, 1] = 0.0
    X_B = Z_B @ mixing_A
    scenarios["B_z1z3z4"] = {"Z": Z_B, "X": X_B, "active": torch.tensor([0, 2, 3])}

    # C: z1,z3 only
    Z_C = copy.deepcopy(Z)
    Z_C[:, :, 1] = 0.0
    Z_C[:, :, 3] = 0.0
    X_C = Z_C @ mixing_A
    scenarios["C_z1z3"] = {"Z": Z_C, "X": X_C, "active": torch.tensor([0, 2])}

    return SyntheticData(Z_full=Z, X_full=X, mixing_A=mixing_A, scenarios=scenarios)


# -----------------------------------------------------------------------------
# MoRe model
# -----------------------------------------------------------------------------


class MoRe(nn.Module):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        layer: int = 1,
        init_f: Optional[torch.Tensor] = None,
        init_B: Optional[torch.Tensor] = None,
        init_grad: bool = False,
        init_dims: Optional[Sequence[int]] = None,
        use_align_T: bool = True,
        align_T_grad: bool = True,
        use_old_gates: bool = True,
        use_new_gates: bool = True,
        gate_init: float = 4.0,
        hierarchy_gate: bool = True,
        freeze_old_structure: bool = True,
        forbid_new_to_old: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.z_dim = int(z_dim)
        self.layer = int(layer)
        self.z_per_layer = max(1, self.z_dim // max(1, self.layer))
        self.use_old_gates = use_old_gates
        self.use_new_gates = use_new_gates
        self.hierarchy_gate = hierarchy_gate
        self.freeze_old_structure = freeze_old_structure
        self.forbid_new_to_old = forbid_new_to_old
        self.eps = eps

        if init_dims is not None:
            init_dims_t = torch.as_tensor(init_dims, dtype=torch.long)
            assert init_dims_t.ndim == 1
            assert int(init_dims_t.min()) >= 0
            assert int(init_dims_t.max()) < self.z_dim
            self.register_buffer("init_dims", init_dims_t)
        else:
            self.init_dims = None

        # f: x -> z
        if init_f is not None:
            assert init_dims is not None
            f_r, f_c = init_f.shape
            assert f_r <= self.x_dim, f"init_f rows {f_r} > x_dim {self.x_dim}"
            assert len(init_dims) == f_c
            self.f_init_shape = (f_r, f_c)
            self.f_init = nn.Parameter(init_f.clone().detach(), requires_grad=init_grad)
            # Small random is more stable than U(0,1) for expanded columns.
            self.f_rest = nn.Parameter(0.05 * torch.randn(size=(self.x_dim, self.z_dim)), requires_grad=True)
            f_init_mask = torch.zeros(size=(self.x_dim, self.z_dim))
            f_init_mask[:f_r, self.init_dims] = 1.0
            self.register_buffer("f_init_mask", f_init_mask)

            if use_align_T:
                self.align_T = nn.Parameter(torch.eye(f_c), requires_grad=align_T_grad)
            else:
                self.register_buffer("align_T", torch.eye(f_c))

            if use_old_gates:
                self.old_gate_logits = nn.Parameter(torch.ones(f_c) * gate_init, requires_grad=True)
            else:
                self.register_buffer("old_gate_logits", torch.ones(f_c) * gate_init)

            old_set = set(int(x) for x in init_dims)
            new_dims = [j for j in range(self.z_dim) if j not in old_set]
            self.register_buffer("new_dims", torch.as_tensor(new_dims, dtype=torch.long))
            if use_new_gates and len(new_dims) > 0:
                self.new_gate_logits = nn.Parameter(torch.ones(len(new_dims)) * gate_init, requires_grad=True)
            else:
                self.new_gate_logits = None
        else:
            self.f_init_shape = None
            self.f = nn.Parameter(0.05 * torch.randn(size=(self.x_dim, self.z_dim)), requires_grad=True)
            self.align_T = None
            self.old_gate_logits = None
            self.new_gate_logits = None
            self.register_buffer("new_dims", torch.arange(self.z_dim, dtype=torch.long))

        # B: z_t0 -> z_t1 dynamics
        if init_B is not None:
            assert init_dims is not None
            B_r, B_c = init_B.shape
            assert B_r == B_c == len(init_dims)
            self.B_init_shape = (B_r, B_c)
            # Old-old structure is normally fixed for stability. If init_grad=True,
            # user explicitly permits old structure modification.
            self.B_init = nn.Parameter(init_B.clone().detach(), requires_grad=(init_grad and not freeze_old_structure))
            self.B_rest = nn.Parameter(0.05 * torch.randn(size=(self.z_dim, self.z_dim)), requires_grad=True)
            B_init_mask = torch.zeros(size=(self.z_dim, self.z_dim))
            B_init_mask[self.init_dims[:, None], self.init_dims[None, :]] = 1.0
            self.register_buffer("B_init_mask", B_init_mask)
            W_hier = init_B.clone().detach().abs()
            W_hier = W_hier / (W_hier.max() + self.eps)
            self.register_buffer("W_hier_old", W_hier)
        else:
            self.B_init_shape = None
            self.B = nn.Parameter(0.05 * torch.randn(size=(self.z_dim, self.z_dim)), requires_grad=True)
            self.W_hier_old = None

    def old_gates(self, tau: float = 1.0) -> Optional[torch.Tensor]:
        if self.old_gate_logits is None:
            return None
        return torch.sigmoid(self.old_gate_logits / tau)

    def new_gates(self, tau: float = 1.0) -> Optional[torch.Tensor]:
        if self.new_gate_logits is None:
            return None
        return torch.sigmoid(self.new_gate_logits / tau)

    def assemble_f(self, tau: float = 1.0) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.f_init_shape is None:
            return self.f, None, None
        f_r, _ = self.f_init_shape
        f_init_full = torch.zeros_like(self.f_rest)
        f_reuse = self.f_init
        if self.align_T is not None:
            f_reuse = f_reuse @ self.align_T
        old_g = self.old_gates(tau=tau)
        if old_g is not None:
            f_reuse = f_reuse * old_g[None, :]
        f_init_full[:f_r, self.init_dims] = f_reuse
        f = self.f_rest * (1.0 - self.f_init_mask) + f_init_full
        new_g = self.new_gates(tau=tau)
        if new_g is not None and len(self.new_dims) > 0:
            f[:, self.new_dims] = f[:, self.new_dims] * new_g[None, :]
        return f, old_g, new_g

    def hierarchy_mask(self, raw_B: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(size=(self.z_dim, self.z_dim), device=raw_B.device)
        if self.layer > 1:
            # B[row output, col input]; zero higher-input -> lower-output.
            for layer_idx in range(self.layer):
                row_start = layer_idx * self.z_per_layer
                row_end = min((layer_idx + 1) * self.z_per_layer, self.z_dim)
                col_start = min((layer_idx + 1) * self.z_per_layer, self.z_dim)
                if col_start < self.z_dim:
                    mask[row_start:row_end, col_start:] = 0.0
        if self.forbid_new_to_old and self.f_init_shape is not None and len(self.new_dims) > 0:
            # Forbid new -> old: input new columns cannot predict output old rows.
            mask[self.init_dims[:, None], self.new_dims[None, :]] = 0.0
        return mask

    def assemble_B(self) -> torch.Tensor:
        if self.B_init_shape is None:
            return self.B
        B_init_full = torch.zeros_like(self.B_rest)
        B_init_full[self.init_dims[:, None], self.init_dims[None, :]] = self.B_init
        return self.B_rest * (1.0 - self.B_init_mask) + B_init_full

    def forward(self, X: torch.Tensor, tau: float = 1.0) -> Dict[str, torch.Tensor | None]:
        f, old_g, new_g = self.assemble_f(tau=tau)
        raw_B = self.assemble_B()
        mask = self.hierarchy_mask(raw_B)
        B = raw_B * mask

        gram = f.T @ f + self.eps * torch.eye(f.shape[1], device=f.device)
        g_inv = torch.linalg.pinv(f)
        Z = X @ f
        est_X = Z @ g_inv
        Z_t0 = Z[:, 0, :]
        Z_t1 = Z[:, 1, :]
        E_t1 = Z_t1 - Z_t0 @ B.T

        loss_recon = F.mse_loss(est_X, X)
        if f.shape[0] == f.shape[1]:
            logdet_penalty = -torch.slogdet(f + self.eps * torch.eye(f.shape[0], device=f.device))[1]
        else:
            logdet_penalty = -0.5 * torch.slogdet(gram)[1]
        loss_indep = E_t1.abs().mean() + logdet_penalty

        loss_sp = torch.tensor(0.0, device=X.device)
        if self.layer > 1:
            denom = 0
            for layer_idx in range(1, self.layer):
                r0 = layer_idx * self.z_per_layer
                r1 = min((layer_idx + 1) * self.z_per_layer, self.z_dim)
                c0 = (layer_idx - 1) * self.z_per_layer
                c1 = min(layer_idx * self.z_per_layer, self.z_dim)
                if r0 < r1 and c0 < c1:
                    loss_sp = loss_sp + B[r0:r1, c0:c1].abs().mean()
                    denom += 1
            loss_sp = loss_sp / max(denom, 1)

        loss_T = torch.tensor(0.0, device=X.device)
        if self.align_T is not None:
            eye = torch.eye(self.align_T.shape[0], device=X.device)
            loss_T = F.mse_loss(self.align_T, eye)

        loss_old_inactive = torch.tensor(0.0, device=X.device)
        if old_g is not None:
            loss_old_inactive = (1.0 - old_g).mean()

        loss_new_active = torch.tensor(0.0, device=X.device)
        if new_g is not None:
            loss_new_active = new_g.mean()

        loss_hier_gate = torch.tensor(0.0, device=X.device)
        if self.hierarchy_gate and old_g is not None and self.W_hier_old is not None:
            gi = old_g[:, None]
            gj = old_g[None, :]
            loss_hier_gate = (self.W_hier_old.to(X.device) * gj * (1.0 - gi)).mean()

        return {
            "loss_recon": loss_recon,
            "loss_indep": loss_indep,
            "loss_sp": loss_sp,
            "loss_T": loss_T,
            "loss_old_inactive": loss_old_inactive,
            "loss_new_active": loss_new_active,
            "loss_hier_gate": loss_hier_gate,
            "B": B,
            "f": f,
            "Z": Z,
            "old_gates": old_g,
            "new_gates": new_g,
            "align_T": self.align_T,
        }


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    name: str
    z_dim: int
    layer: int
    lr: float = 0.01
    epochs: int = 20000
    w_indep: float = 1.0
    w_sp: float = 0.0
    w_T: float = 0.005
    w_old_inactive: float = 0.001
    w_new_active: float = 0.001
    w_hier_gate: float = 0.001
    init_grad: bool = False
    use_align_T: bool = True
    align_T_grad: bool = True
    use_old_gates: bool = True
    use_new_gates: bool = True
    hierarchy_gate: bool = True
    freeze_old_structure: bool = True
    forbid_new_to_old: bool = True
    gate_init: float = 4.0
    tau_start: float = 1.0
    tau_end: float = 0.1


def train_more(
    X: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    init_f: Optional[torch.Tensor] = None,
    init_B: Optional[torch.Tensor] = None,
    init_dims: Optional[Sequence[int]] = None,
    verbose_every: int = 0,
) -> Tuple[MoRe, Dict[str, Any]]:
    X = X.to(device)
    model = MoRe(
        x_dim=X.shape[-1],
        z_dim=cfg.z_dim,
        layer=cfg.layer,
        init_f=init_f.to(device) if init_f is not None else None,
        init_B=init_B.to(device) if init_B is not None else None,
        init_grad=cfg.init_grad,
        init_dims=init_dims,
        use_align_T=cfg.use_align_T,
        align_T_grad=cfg.align_T_grad,
        use_old_gates=cfg.use_old_gates,
        use_new_gates=cfg.use_new_gates,
        gate_init=cfg.gate_init,
        hierarchy_gate=cfg.hierarchy_gate,
        freeze_old_structure=cfg.freeze_old_structure,
        forbid_new_to_old=cfg.forbid_new_to_old,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    t0 = time.perf_counter()
    last_out = None
    for epoch in range(cfg.epochs):
        frac = epoch / max(cfg.epochs - 1, 1)
        tau = cfg.tau_start * (cfg.tau_end / cfg.tau_start) ** frac
        out = model(X, tau=tau)
        loss = (
            0.0 * out["loss_recon"]
            + cfg.w_indep * out["loss_indep"]
            + cfg.w_sp * out["loss_sp"]
            + cfg.w_T * out["loss_T"]
            + cfg.w_old_inactive * out["loss_old_inactive"]
            + cfg.w_new_active * out["loss_new_active"]
            + cfg.w_hier_gate * out["loss_hier_gate"]
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_out = out
        if verbose_every and (epoch + 1) % verbose_every == 0:
            msg = (
                f"[{cfg.name}] epoch {epoch+1}/{cfg.epochs} loss={loss.item():.4f} "
                f"indep={out['loss_indep'].item():.4f} T={out['loss_T'].item():.4f} "
                f"old_inact={out['loss_old_inactive'].item():.4f} new_act={out['loss_new_active'].item():.4f}"
            )
            if out["old_gates"] is not None:
                msg += f" old_g={to_cpu_np(out['old_gates']).round(3).tolist()}"
            if out["new_gates"] is not None:
                msg += f" new_g={to_cpu_np(out['new_gates']).round(3).tolist()}"
            print(msg, flush=True)
    train_time = time.perf_counter() - t0
    with torch.no_grad():
        out = model(X, tau=cfg.tau_end)
    info = {
        "out": {k: (v.detach().cpu() if torch.is_tensor(v) else None) for k, v in out.items()},
        "train_time_sec": train_time,
        **count_parameters(model),
    }
    return model, info


def estimate_Z(model: MoRe, X: torch.Tensor, device: torch.device, tau: float = 0.1) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(X.to(device), tau=tau)
        return out["Z"].detach().cpu()


def corr_abs_matrix(Z_hat: torch.Tensor, Z_true: torch.Tensor) -> torch.Tensor:
    A = Z_hat.reshape(-1, Z_hat.shape[-1]).float()
    B = Z_true.reshape(-1, Z_true.shape[-1]).float()
    A = A - A.mean(dim=0, keepdim=True)
    B = B - B.mean(dim=0, keepdim=True)
    A = A / (A.std(dim=0, keepdim=True) + 1e-8)
    B = B / (B.std(dim=0, keepdim=True) + 1e-8)
    return ((A.T @ B) / max(A.shape[0] - 1, 1)).abs()


def best_mcc(Z_hat: torch.Tensor, Z_true: torch.Tensor, true_indices: Sequence[int]) -> Tuple[float, List[Tuple[int, int, float]], torch.Tensor]:
    Zt = Z_true[:, :, list(true_indices)]
    C = corr_abs_matrix(Z_hat, Zt)
    n_hat, n_true = C.shape
    if n_true == 0:
        return float("nan"), [], C
    # Brute force over estimated dimensions for small experiments.
    best_score = -1.0
    best_pairs: List[Tuple[int, int, float]] = []
    for est_perm in permutations(range(n_hat), min(n_hat, n_true)):
        vals = [float(C[est_perm[j], j].item()) for j in range(len(est_perm))]
        score = float(np.mean(vals))
        if score > best_score:
            best_score = score
            best_pairs = [(int(est_perm[j]), int(true_indices[j]), vals[j]) for j in range(len(est_perm))]
    return best_score, best_pairs, C


def old_module_gt_mapping(Z_old_hat: torch.Tensor, Z_old_true: torch.Tensor, old_true_indices: Sequence[int]) -> Dict[int, int]:
    """Map old module index -> GT factor index by best absolute correlation."""
    _, pairs, _ = best_mcc(Z_old_hat, Z_old_true, old_true_indices)
    return {est_idx: gt_idx for est_idx, gt_idx, _ in pairs}


def gate_correctness(
    old_gates: Optional[torch.Tensor],
    old_module_to_gt: Dict[int, int],
    target_active_gt: Sequence[int],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    if old_gates is None:
        return {
            "gate_accuracy": float("nan"),
            "gate_tp": 0,
            "gate_tn": 0,
            "gate_fp": 0,
            "gate_fn": 0,
            "old_active_count": float("nan"),
            "expected_old_active_count": float("nan"),
        }
    g = old_gates.detach().cpu()
    target_set = set(int(x) for x in target_active_gt)
    y_true = []
    y_pred = []
    for old_idx, gt_idx in old_module_to_gt.items():
        y_true.append(1 if gt_idx in target_set else 0)
        y_pred.append(1 if float(g[old_idx]) > threshold else 0)
    if len(y_true) == 0:
        acc = float("nan")
    else:
        acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {
        "gate_accuracy": acc,
        "gate_tp": tp,
        "gate_tn": tn,
        "gate_fp": fp,
        "gate_fn": fn,
        "old_active_count": int((g > threshold).sum().item()),
        "expected_old_active_count": int(sum(y_true)),
    }


def save_model_parts(
    out_dir: Path,
    tag: str,
    model: MoRe,
    basis: Basis,
    train_info: Dict[str, Any],
    cfg: TrainConfig,
) -> None:
    d = ensure_dir(out_dir / tag)
    with torch.no_grad():
        out = train_info["out"]
        payload = {
            "model_state_dict": model.state_dict(),
            "f": out["f"],
            "B": out["B"],
            "old_gates": out["old_gates"],
            "new_gates": out["new_gates"],
            "align_T": out["align_T"],
            "basis_mean": basis.mean.detach().cpu(),
            "basis_components": basis.components.detach().cpu(),
            "config": asdict(cfg),
        }
    torch.save(payload, d / "parts.pt")


# -----------------------------------------------------------------------------
# Experiment definitions
# -----------------------------------------------------------------------------


def make_transition_configs(
    old_dim: int,
    target_dim: int,
    layer: int,
    epochs: int,
    lr: float,
    w_old_inactive: float,
    w_new_active: float,
    w_hier_gate: float,
) -> List[TrainConfig]:
    """Baselines and ablations for one transition."""
    z_dim = target_dim
    base = dict(z_dim=z_dim, layer=layer, lr=lr, epochs=epochs, w_indep=1.0, w_sp=0.0)
    return [
        TrainConfig(name="scratch", **base, use_align_T=False, use_old_gates=False, use_new_gates=False, hierarchy_gate=False),
        TrainConfig(name="finetune", **base, init_grad=True, use_align_T=False, use_old_gates=False, use_new_gates=False, hierarchy_gate=False, freeze_old_structure=False, forbid_new_to_old=False),
        TrainConfig(name="expansion_only", **base, init_grad=False, use_align_T=False, use_old_gates=False, use_new_gates=False, hierarchy_gate=False),
        TrainConfig(name="reuse_T", **base, init_grad=False, use_align_T=True, use_old_gates=False, use_new_gates=False, hierarchy_gate=False, w_T=0.005),
        TrainConfig(name="selective_no_T", **base, init_grad=False, use_align_T=False, use_old_gates=True, use_new_gates=True, hierarchy_gate=True, w_old_inactive=w_old_inactive, w_new_active=w_new_active, w_hier_gate=w_hier_gate),
        TrainConfig(name="selective_no_new_gate", **base, init_grad=False, use_align_T=True, use_old_gates=True, use_new_gates=False, hierarchy_gate=True, w_old_inactive=w_old_inactive, w_new_active=0.0, w_hier_gate=w_hier_gate, w_T=0.005),
        TrainConfig(name="selective_no_hier", **base, init_grad=False, use_align_T=True, use_old_gates=True, use_new_gates=True, hierarchy_gate=False, w_old_inactive=w_old_inactive, w_new_active=w_new_active, w_hier_gate=0.0, w_T=0.005),
        TrainConfig(name="MoRe_full", **base, init_grad=False, use_align_T=True, use_old_gates=True, use_new_gates=True, hierarchy_gate=True, w_old_inactive=w_old_inactive, w_new_active=w_new_active, w_hier_gate=w_hier_gate, w_T=0.005),
    ]


def run_one_transition(
    transition_name: str,
    old_name: str,
    target_name: str,
    old_active: Sequence[int],
    target_active: Sequence[int],
    data: SyntheticData,
    out_dir: Path,
    device: torch.device,
    seed: int,
    epochs: int,
    lr: float,
    old_basis_dim: int,
    k_extra: int,
    target_total_dim: int,
    old_layer: int,
    target_layer: int,
    w_old_inactive: float,
    w_new_active: float,
    w_hier_gate: float,
    verbose_every: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    old = data.scenarios[old_name]
    target = data.scenarios[target_name]

    # Learn old model/library.
    basis_old = fit_pca_basis(old["X"], k=old_basis_dim)
    X_old_coord = transform_basis(old["X"], basis_old)
    old_cfg = TrainConfig(name=f"{transition_name}_old_library", z_dim=old_basis_dim, layer=old_layer, lr=lr, epochs=epochs, use_align_T=False, use_old_gates=False, use_new_gates=False, hierarchy_gate=False)
    old_model, old_info = train_more(X_old_coord, old_cfg, device=device, verbose_every=verbose_every)
    Z_old_hat = estimate_Z(old_model, X_old_coord, device=device)
    old_mcc, old_pairs, old_corr = best_mcc(Z_old_hat, old["Z"], old_active)
    old_map = old_module_gt_mapping(Z_old_hat, old["Z"], old_active)
    save_model_parts(out_dir, f"seed{seed}_{transition_name}_old_library", old_model, basis_old, old_info, old_cfg)

    # Expand observation coordinates with residual candidates, then transform target and old data into expanded basis.
    basis_target, V_new = expand_basis_with_residual(target["X"], basis_old, k_extra=k_extra)
    X_target_coord = transform_basis(target["X"], basis_target)
    X_old_in_target_coord = transform_basis(old["X"], basis_target)

    init_f = old_info["out"]["f"]
    init_B = old_info["out"]["B"]
    init_dims = list(range(old_basis_dim))

    configs = make_transition_configs(
        old_dim=old_basis_dim,
        target_dim=target_total_dim,
        layer=target_layer,
        epochs=epochs,
        lr=lr,
        w_old_inactive=w_old_inactive,
        w_new_active=w_new_active,
        w_hier_gate=w_hier_gate,
    )

    for cfg in configs:
        print(f"\n=== seed={seed} transition={transition_name} method={cfg.name} ===", flush=True)
        if cfg.name == "scratch":
            model, info = train_more(X_target_coord, cfg, device=device, verbose_every=verbose_every)
        else:
            model, info = train_more(
                X_target_coord,
                cfg,
                device=device,
                init_f=init_f,
                init_B=init_B,
                init_dims=init_dims,
                verbose_every=verbose_every,
            )
        Z_target_hat = estimate_Z(model, X_target_coord, device=device)
        target_mcc, target_pairs, target_corr = best_mcc(Z_target_hat, target["Z"], target_active)

        # Stability/backward compatibility: evaluate adapted model on old data in expanded target coordinates.
        Z_old_with_adapted = estimate_Z(model, X_old_in_target_coord, device=device)
        old_after_mcc, old_after_pairs, old_after_corr = best_mcc(Z_old_with_adapted, old["Z"], old_active)
        forgetting = old_mcc - old_after_mcc

        old_g = info["out"].get("old_gates", None)
        new_g = info["out"].get("new_gates", None)
        gate_stats = gate_correctness(old_g, old_map, target_active)
        row = {
            "seed": seed,
            "transition": transition_name,
            "method": cfg.name,
            "old_name": old_name,
            "target_name": target_name,
            "old_active_gt": list(map(int, old_active)),
            "target_active_gt": list(map(int, target_active)),
            "old_library_mcc": old_mcc,
            "target_mcc_plasticity": target_mcc,
            "old_after_adapt_mcc_backward_compat": old_after_mcc,
            "forgetting_backward_compat": forgetting,
            "old_library_pairs": old_pairs,
            "target_pairs": target_pairs,
            "old_after_pairs": old_after_pairs,
            "old_gates": None if old_g is None else [float(x) for x in old_g.detach().cpu().flatten()],
            "new_gates": None if new_g is None else [float(x) for x in new_g.detach().cpu().flatten()],
            "new_active_count": float("nan") if new_g is None else int((new_g > 0.5).sum().item()),
            "target_total_dim": target_total_dim,
            "old_basis_dim": old_basis_dim,
            "k_extra": k_extra,
            "basis_dim": basis_target.dim,
            "train_time_sec": info["train_time_sec"],
            **{k: v for k, v in info.items() if k.endswith("params") or k.startswith("trainable_")},
            **gate_stats,
        }
        rows.append(row)

        # Save proposed method and all ablations/baselines for inspection.
        save_model_parts(out_dir, f"seed{seed}_{transition_name}_{cfg.name}", model, basis_target, info, cfg)
        torch.save({
            "target_corr": target_corr,
            "old_after_corr": old_after_corr,
            "old_library_corr": old_corr,
            "target_pairs": target_pairs,
            "old_after_pairs": old_after_pairs,
            "old_library_pairs": old_pairs,
            "old_module_to_gt": old_map,
        }, out_dir / f"seed{seed}_{transition_name}_{cfg.name}_eval.pt")

    return rows


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    metrics_csv = out_dir / "metrics.csv"
    if not rows:
        return
    # Flatten non-scalar values as JSON strings.
    keys = sorted({k for r in rows for k in r.keys()})
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            rr = {}
            for k in keys:
                v = r.get(k, None)
                if isinstance(v, (list, tuple, dict)):
                    rr[k] = json.dumps(v)
                elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    rr[k] = "nan"
                else:
                    rr[k] = v
            writer.writerow(rr)

    # Summary by transition/method.
    summary: Dict[str, Any] = {}
    for transition in sorted(set(r["transition"] for r in rows)):
        summary[transition] = {}
        for method in sorted(set(r["method"] for r in rows if r["transition"] == transition)):
            subset = [r for r in rows if r["transition"] == transition and r["method"] == method]
            metrics = [
                "target_mcc_plasticity",
                "old_after_adapt_mcc_backward_compat",
                "forgetting_backward_compat",
                "gate_accuracy",
                "old_active_count",
                "new_active_count",
                "train_time_sec",
                "trainable_params",
                "total_params",
            ]
            summary[transition][method] = {}
            for m in metrics:
                vals = []
                for r in subset:
                    try:
                        val = float(r.get(m, float("nan")))
                    except Exception:
                        val = float("nan")
                    if not math.isnan(val):
                        vals.append(val)
                if vals:
                    summary[transition][method][m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                else:
                    summary[transition][method][m] = {"mean": None, "std": None}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoRe selective adaptation synthetic experiment")
    parser.add_argument("--out_dir", type=str, default="more_results")
    parser.add_argument("--T", type=int, default=50000)
    parser.add_argument("--obs_dim", type=int, default=4)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose_every", type=int, default=0)
    parser.add_argument("--w_old_inactive", type=float, default=0.001)
    parser.add_argument("--w_new_active", type=float, default=0.001)
    parser.add_argument("--w_hier_gate", type=float, default=0.001)
    parser.add_argument("--skip_B_to_C", action="store_true")
    parser.add_argument("--skip_A_to_B", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    device = torch.device(args.device)
    all_rows: List[Dict[str, Any]] = []
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    for seed in args.seeds:
        print(f"\n############################ seed={seed} ############################", flush=True)
        data = generate_base_data(T=args.T, noise_scale=args.noise_scale, obs_dim=args.obs_dim, seed=seed)
        torch.save({
            "mixing_A": data.mixing_A,
            "Z_full_shape": tuple(data.Z_full.shape),
            "X_full_shape": tuple(data.X_full.shape),
        }, out_dir / f"seed{seed}_data_info.pt")

        if not args.skip_A_to_B:
            rows = run_one_transition(
                transition_name="A_z1z2_to_B_z1z3z4",
                old_name="A_z1z2",
                target_name="B_z1z3z4",
                old_active=[0, 1],
                target_active=[0, 2, 3],
                data=data,
                out_dir=out_dir,
                device=device,
                seed=seed,
                epochs=args.epochs,
                lr=args.lr,
                old_basis_dim=2,
                k_extra=2,
                target_total_dim=4,
                old_layer=1,
                target_layer=2,
                w_old_inactive=args.w_old_inactive,
                w_new_active=args.w_new_active,
                w_hier_gate=args.w_hier_gate,
                verbose_every=args.verbose_every,
            )
            all_rows.extend(rows)
            write_outputs(all_rows, out_dir)

        if not args.skip_B_to_C:
            rows = run_one_transition(
                transition_name="B_z1z3z4_to_C_z1z3",
                old_name="B_z1z3z4",
                target_name="C_z1z3",
                old_active=[0, 2, 3],
                target_active=[0, 2],
                data=data,
                out_dir=out_dir,
                device=device,
                seed=seed,
                epochs=args.epochs,
                lr=args.lr,
                old_basis_dim=4,
                k_extra=0,
                target_total_dim=4,
                old_layer=2,
                target_layer=2,
                w_old_inactive=args.w_old_inactive,
                w_new_active=args.w_new_active,
                w_hier_gate=args.w_hier_gate,
                verbose_every=args.verbose_every,
            )
            all_rows.extend(rows)
            write_outputs(all_rows, out_dir)

    write_outputs(all_rows, out_dir)
    print(f"\nDone. Results saved to: {out_dir.resolve()}", flush=True)
    print(f"Metrics CSV: {out_dir / 'metrics.csv'}", flush=True)
    print(f"Summary JSON: {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
