"""
more_core.py — MoRe model core: shared by all experiment scripts.

Pipeline (one adaptation step):
  Step 1. Coordinate projection:   X_proj = X @ V.T          (N,T,d_obs) -> (N,T,d_coord)
  Step 2. Inference with old model: Z_temp = X_proj @ F       (N,T,d_coord) -> (N,T,m)
           E_temp = Z_t1 - Z_t0 @ B.T
  Step 3. Self-consistency score a_j (static, analytic):
           From H1 = corr(E,Z_t0), H2 = corr(E,Z_t1), H3 = corr(Z_t0,Z_t1)
           a_j = (1-|H1[j,j]|)*(1-max_{i≠j}|H1[i,j]|)*|H2[j,j]|*|H3[j,j]|*(1-max_{i≠j}|H3[i,j]|)
           Candidate new modules get a_j = 0 by construction.
  Step 4. Expand coordinates: R = X - X_proj @ V, PCA(R) -> V_new; extend B.
  Step 5. Learn discrete gates via Hard Concrete (independent Bernoulli relaxation):
           w_j = a_j * g_j,   g_j ~ BinaryConcrete
           Z = [X_proj @ F * w_old * T_align | X @ V_new @ F_new * w_new]
           E = Z_t1 - Z_t0 @ B_extended.T
  Step 6. Losses:
           L_data  = ||E||_1 - logdet(F_active)
           L_align = ||T - I||^2
           L_sp    = sum_j w_j              (sparsity on gates)
           L_B_sp  = ||B_dd||_1 + ||B_d||_1 (sparsity on new B blocks)
           L_B_reg = ||B_init - B[:m,:m]||^2 (keep old dynamics stable)
  Step 7. Update: store gate assignment, append V_new, append new modules.
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field, asdict
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def to_np(x: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    return None if x is None else x.detach().cpu().numpy()

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Data generation — 7 scenarios with data1 = z1,z2
# =============================================================================

@dataclass
class SyntheticData:
    Z_full: torch.Tensor
    X_full: torch.Tensor
    mixing_A: torch.Tensor
    scenarios: Dict[str, Dict[str, torch.Tensor]]


def generate_data(T: int, obs_dim: int, noise_scale: float, seed: int) -> SyntheticData:
    """
    Generates sequential latent data with 4 factors z1..z4 and linear mixing.
    data1 (old) = A_z1z2.
    7 data2 scenarios cover all selection cases from theory.
    """
    set_seed(seed)
    lap = torch.distributions.Laplace(0.0, noise_scale)

    def sample(n): return lap.sample((T, 2, n))  # (T, 2, n) = (T, [t0,t1], n)

    E1, E2, E3, E4 = sample(1), sample(1), sample(1), sample(1)

    # Layer 1: z1, z2 (contemporaneous + lagged within-layer dependencies).
    Z1 = E1.clone()
    Z1[:, 1, 0] = 0.8*E1[:, 0, 0] + E1[:, 1, 0] + 0.5*E2[:, 0, 0]
    Z2 = E2.clone()
    Z2[:, 1, 0] = 0.8*E2[:, 0, 0] + E2[:, 1, 0] + 0.5*E1[:, 0, 0]

    # Layer 2: z3, z4 (driven by layer 1 with time delay — hierarchy).
    Z3 = E3.clone()
    Z3[:, 1, 0] = (0.8*E3[:, 0, 0] + 0.6*Z1[:, 0, 0]
                   + 0.8*Z2[:, 0, 0] + E3[:, 1, 0] + 0.5*E4[:, 0, 0])
    Z4 = E4.clone()
    Z4[:, 1, 0] = (0.8*E4[:, 0, 0] + 0.4*Z1[:, 0, 0]
                   + 0.6*Z2[:, 0, 0] + E4[:, 1, 0] + 0.5*E3[:, 0, 0])

    # Z: (T, 2, 4)
    Z = torch.cat([Z1, Z2, Z3, Z4], dim=-1)
    mixing_A = torch.rand(4, obs_dim)
    X = Z @ mixing_A  # (T, 2, obs_dim)

    def make(mask_indices_zero):
        Zm = Z.clone()
        for i in mask_indices_zero:
            Zm[:, :, i] = 0.0
        Xm = Zm @ mixing_A
        active = [j for j in range(4) if j not in mask_indices_zero]
        return {"Z": Zm, "X": Xm, "active": torch.tensor(active)}

    scenarios = {
        # data1
        "A_z1z2":      make([2, 3]),
        # 7 data2 cases
        "D2_z1z2":     make([2, 3]),          # Case 1: same as data1
        "D2_z1z2z3z4": make([]),              # Case 2: full reuse + expand
        "D2_z3z4":     make([0, 1]),          # Case 3: drop all + expand
        "B_z1z3z4":    make([1]),             # Case 4: keep z1 drop z2 expand
        "D2_z2z3z4":   make([0]),             # Case 5: keep z2 drop z1 expand
        "D2_z1":       make([1, 2, 3]),       # Case 6: keep z1 only
        "D2_z2":       make([0, 2, 3]),       # Case 7: keep z2 only
        # secondary (for B->C compatibility)
        "C_z1z3":      make([1, 3]),
    }
    return SyntheticData(Z_full=Z, X_full=X, mixing_A=mixing_A, scenarios=scenarios)


# All 7 transitions: (name, old_scenario, target_scenario, old_active, target_active,
#                     old_basis_dim, k_extra, target_total_dim, old_layer, target_layer)
ALL_TRANSITIONS = [
    ("case1_z1z2_same",  "A_z1z2","D2_z1z2",    [0,1],[0,1],     2,0,2,1,1),
    ("case2_z1z2z3z4",   "A_z1z2","D2_z1z2z3z4",[0,1],[0,1,2,3], 2,2,4,1,2),
    ("case3_z3z4",       "A_z1z2","D2_z3z4",     [0,1],[2,3],     2,2,4,1,1),
    ("case4_z1z3z4",     "A_z1z2","B_z1z3z4",    [0,1],[0,2,3],   2,2,4,1,2),
    ("case5_z2z3z4",     "A_z1z2","D2_z2z3z4",   [0,1],[1,2,3],   2,2,4,1,2),
    ("case6_z1_only",    "A_z1z2","D2_z1",        [0,1],[0],       2,0,2,1,1),
    ("case7_z2_only",    "A_z1z2","D2_z2",        [0,1],[1],       2,0,2,1,1),
]


# =============================================================================
# Step 1-2: Coordinate system and projection
# =============================================================================

@dataclass
class CoordBasis:
    """Observation coordinate system V: maps X -> X_proj = X @ V."""
    V: torch.Tensor        # (obs_dim, coord_dim)
    mean: torch.Tensor     # (obs_dim,)  for centering

    @property
    def coord_dim(self): return self.V.shape[1]
    @property
    def obs_dim(self): return self.V.shape[0]

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """X: (..., obs_dim) -> (..., coord_dim)"""
        return (X - self.mean) @ self.V

    def reconstruct(self, X_proj: torch.Tensor) -> torch.Tensor:
        """X_proj: (..., coord_dim) -> (..., obs_dim)"""
        return X_proj @ self.V.T + self.mean

    def to(self, device) -> "CoordBasis":
        return CoordBasis(self.V.to(device), self.mean.to(device))


def fit_pca_basis(X: torch.Tensor, k: int) -> CoordBasis:
    """Fit PCA basis of rank k from X (T,2,obs_dim)."""
    Xf = X.reshape(-1, X.shape[-1])
    mean = Xf.mean(0)
    _, _, Vh = torch.linalg.svd(Xf - mean, full_matrices=False)
    return CoordBasis(V=Vh[:k].T.contiguous(), mean=mean)


def expand_basis(X_new: torch.Tensor, basis: CoordBasis, k_extra: int,
                 eps: float = 1e-8) -> Tuple[CoordBasis, torch.Tensor]:
    """
    Step 3: Expand coordinate basis with residual directions.
    Returns extended basis and V_new (obs_dim, k_extra) for new modules.
    """
    if k_extra <= 0:
        return basis, torch.zeros(basis.obs_dim, 0)

    # Residual in observation space after projecting through existing basis.
    X_proj = basis.project(X_new)
    R = X_new - basis.reconstruct(X_proj)   # (..., obs_dim)
    Rf = R.reshape(-1, R.shape[-1])
    Rc = Rf - Rf.mean(0)
    try:
        _, _, Vh = torch.linalg.svd(Rc, full_matrices=False)
    except RuntimeError:
        _, _, Vh = torch.linalg.svd(Rc.cpu(), full_matrices=False)
        Vh = Vh.to(Rc.device)

    V_cand = Vh[:k_extra].T  # (obs_dim, k_extra)
    # Orthogonalize against existing basis.
    V_cand = V_cand - basis.V @ (basis.V.T @ V_cand)
    Q, R_qr = torch.linalg.qr(V_cand, mode="reduced")
    keep = Q[:, torch.abs(torch.diag(R_qr)) > eps]
    if keep.shape[1] < k_extra:
        pad = torch.randn(basis.obs_dim, k_extra - keep.shape[1], device=V_cand.device)
        pad = pad - basis.V @ (basis.V.T @ pad)
        if keep.shape[1] > 0:
            pad = pad - keep @ (keep.T @ pad)
        Q2, _ = torch.linalg.qr(pad, mode="reduced")
        keep = torch.cat([keep, Q2], dim=1)
    V_new = keep[:, :k_extra]
    V_extended = torch.cat([basis.V.to(V_new.device), V_new], dim=1)
    new_basis = CoordBasis(V=V_extended, mean=basis.mean.to(V_new.device))
    return new_basis, V_new


# =============================================================================
# Step 2-3: Analytic self-consistency score
# =============================================================================

def _corr(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pearson correlation matrix: (N,da) x (N,db) -> (da,db)."""
    A = A - A.mean(0); B = B - B.mean(0)
    A = A / A.std(0).clamp(min=eps)
    B = B / B.std(0).clamp(min=eps)
    return (A.T @ B) / max(A.shape[0] - 1, 1)


@torch.no_grad()
def compute_self_consistency(
    X_proj: torch.Tensor,       # (T, 2, coord_dim) — new data in old coordinates
    F: torch.Tensor,            # (coord_dim, m)    — old encoder
    B: torch.Tensor,            # (m, m)            — old dynamics (used only for E)
    F_ref: Optional[torch.Tensor] = None,       # unused, kept for API compat
    X_proj_ref: Optional[torch.Tensor] = None,  # unused, kept for API compat
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute self-consistency score a_j for each old module j.

    Core insight: if factor j is PRESENT in new data, then Z_j = X@F[:,j]
    will be temporally autocorrelated: corr(Z_j_t0, Z_j_t1) will be high.
    If factor j is ABSENT, Z_j is noise: corr(Z_j_t0, Z_j_t1) ≈ 0.

    This is the primary signal. We use TWO quantities:

      S_keep[j] = |corr(Z_j_t0, Z_j_t1)|   (raw temporal autocorrelation)
                  High (>0.3) = factor j is present and temporally structured.
                  Low (<0.1)  = factor j is absent, Z_j is noise.

      S_drop[j] = mean_i |corr(E_i_t1, Z_j_t0)|  (H1 off-diagonal column j)
                  High = j's past drives residuals in other modules = misfit.
                  Low  = j fits cleanly into the new data dynamics.

      a_j = S_keep[j] * (1 - S_drop[j])

    NOTE: We do NOT normalize by H3_ref (old data autocorrelation) because
    H3_ref is also near-zero when OldModel.B collapses (VAR underpowered).
    The raw H3_diagonal is sufficient — it has a natural [0,1] scale.

    Expected output:
      Present module:  a_j ≈ 0.3 - 0.9
      Absent module:   a_j ≈ 0.0 - 0.1
    """
    m = F.shape[1]
    Z_t0 = X_proj[:, 0, :] @ F   # (T, m)
    Z_t1 = X_proj[:, 1, :] @ F   # (T, m)
    E_t1 = Z_t1 - Z_t0 @ B.T    # (T, m)  VAR residuals

    # H1[i,j] = corr(E_i_t1, Z_j_t0) — autoregressive misfit matrix
    H1 = _corr(E_t1, Z_t0)   # (m, m)

    # H3[j,j] = corr(Z_j_t0, Z_j_t1) — raw temporal autocorrelation
    H3 = _corr(Z_t0, Z_t1)   # (m, m)

    # S_keep: raw temporal autocorrelation diagonal.
    # Present factor: strongly autocorrelated (0.3-0.9).
    # Absent factor:  noise, near-zero autocorrelation.
    S_keep = H3.abs().diag().clamp(0.0, 1.0)   # (m,)

    # S_drop: off-diagonal H1 column mean.
    # High = j's past leaks into other modules' residuals = absent/misfit.
    H1_abs = H1.abs()
    H1_offdiag = H1_abs.clone()
    H1_offdiag.fill_diagonal_(0.0)
    S_drop = H1_offdiag.mean(dim=0)   # (m,)

    a = S_keep * (1.0 - S_drop).clamp(0.0, 1.0)

    diag = {
        "H1": H1.cpu(), "H3": H3.cpu(),
        "S_keep": S_keep.cpu(), "S_drop": S_drop.cpu(), "a": a.cpu(),
        "H3_diag": H3.diag().cpu(),
    }
    return a, diag


def analytic_gate_init_logits(
    a_old: torch.Tensor,
    gate_C: float = 3.0,
    noise_floor: float = 0.15,
    temperature: float = 0.5,
) -> Tuple[torch.Tensor, float]:
    """
    Convert self-consistency scores to (old gate logits, new gate init logit).

    Design:
      1. Softmax over a_old (temperature-scaled) gives relative ordering — scale invariant.
         Amplifies differences: [0.6, 0.1] -> [0.73, 0.27]; [0.6, 0.6] -> [0.5, 0.5].

      2. a_magnitude = mean(a_old) captures absolute signal — are ANY modules
         self-consistent? Near zero = nothing fits new data (drop all or full expand).
         High = at least some modules are self-consistent (selective reuse).

      3. Gate logit[j] = gate_C * (2 * softmax_j - 1) * clip(a_magnitude / noise_floor, 0, 1)
         - softmax term: +gate_C for most consistent, -gate_C for least consistent.
         - magnitude gate: when a_magnitude << noise_floor, suppresses all logits
           toward 0 (uncertain), letting the optimizer decide from data.
         - When a_magnitude >> noise_floor, full ±gate_C logits are used.

      4. New gate init = gate_C * clip(1 - a_magnitude / noise_floor, -1, 1)
         - High a_magnitude (old modules fit well): new gates start negative (not needed).
         - Low a_magnitude (old modules don't fit): new gates start positive (expand freely).
         This is the natural complement of old gate confidence.

    Cases:
      Selective reuse [0.6, 0.05]: a_mag=0.33 >> floor → logits=[+C, -C] ✓
      Full reuse      [0.6, 0.55]: a_mag=0.58 >> floor → logits=[+C*0.2, -C*0.2] ≈ neutral ✓
                                    (both near 0.5 softmax → small logits → optimizer decides)
      Drop all        [0.05, 0.03]: a_mag=0.04 < floor → logits≈[0, 0], new_init=+C ✓
      Keep one only   [0.7, 0.02]: a_mag=0.36 >> floor → logits=[+C, -C] ✓

    Returns:
        gate_logits_old: (m,) tensor
        gate_logit_new:  float — scalar init for all new module gates
    """
    a = a_old.float()
    m = len(a)

    # Softmax with temperature (lower temp = sharper separation).
    a_soft = torch.softmax(a / temperature, dim=0)   # (m,) sums to 1

    # Absolute magnitude: how self-consistent are the old modules overall?
    a_magnitude = float(a.mean().item())

    # Magnitude confidence: 0 when nothing fits, 1 when clearly self-consistent.
    mag_confidence = float(min(a_magnitude / max(noise_floor, 1e-8), 1.0))

    # Old gate logits: softmax drives direction, magnitude drives strength.
    # a_soft[j] in [1/m, ~1]; (2*a_soft[j]-1) maps [0,1] -> [-1,1] (centered).
    gate_logits_old = gate_C * (2.0 * a_soft - 1.0) * mag_confidence

    # New gate init: always +2.0 so new modules start on and F_new gets gradient.
    gate_logit_new = 2.0

    return gate_logits_old, gate_logit_new

class HardConcreteGate(nn.Module):
    """
    Hard Concrete / Binary Concrete gate for independent Bernoulli variables.

    Forward (training): sample from stretched distribution, hard-clip to [0,1].
    Forward (eval/tau->0): hard threshold at 0.5 (binary).

    Why not Gumbel-softmax: Gumbel is for categorical (one-of-K). Gates here
    are independent Bernoullis — each is a separate binary decision. Hard
    Concrete is the correct relaxation for this case.

    Reference: Louizos et al. "Learning Sparse Neural Networks through L0 Regularization" ICLR 2018.
    """
    zeta: float = 1.1    # stretch upper bound (> 1)
    gamma: float = -0.1  # stretch lower bound (< 0)

    def __init__(self, n: int, init_logit: float = 0.0):
        super().__init__()
        self.logits = nn.Parameter(torch.full((n,), init_logit))

    def forward(self, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Returns gate values in [0,1].
        During training (hard=False): differentiable soft sample.
        During eval (hard=True): deterministic hard 0/1.
        """
        if hard or not self.training:
            return (torch.sigmoid(self.logits) > 0.5).float()

        # Stretch sigmoid sample through (gamma, zeta) then clamp.
        u = torch.zeros_like(self.logits).uniform_().clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.logits) / tau)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return s_bar.clamp(0.0, 1.0)

    def l0_loss(self, tau: float = 1.0) -> torch.Tensor:
        """
        Expected number of active gates — use as sparsity loss.
        = sum_j sigmoid(logit_j - beta * log(-gamma/zeta))
        where beta = tau for the stretched Concrete.
        """
        beta = - math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.logits - tau * beta).sum()

    @property
    def n(self): return self.logits.shape[0]


# =============================================================================
# MoRe model
# =============================================================================

class MoRe(nn.Module):
    """
    Linear MoRe model for one adaptation step.

    Parameters
    ----------
    coord_dim : int        dimension of projected coordinates (old + new)
    m_old : int            number of old modules
    m_new : int            number of candidate new modules
    a_old : Tensor (m_old,)  static self-consistency scores for old modules
    init_F : Tensor (coord_dim_old, m_old)   frozen old encoder columns
    init_B : Tensor (m_old, m_old)           frozen old dynamics (normally not updated)
    gate_init_logit : float   initial logit for all gates
    gate_logits_override : optional (m_old,)  analytic-derived logit init for old gates
    use_align_T : bool    learn alignment matrix T (m_old x m_old)
    freeze_B_old : bool   keep old B block frozen
    layer_old, layer_new  for hierarchy mask on B
    """

    def __init__(
        self,
        coord_dim_old: int,
        coord_dim_new: int,
        m_old: int,
        m_new: int,
        a_old: torch.Tensor,              # (m_old,) static scores, fixed
        init_F: Optional[torch.Tensor],   # (coord_dim_old, m_old)
        init_B: Optional[torch.Tensor],   # (m_old, m_old)
        gate_init_logit: float = 0.0,
        gate_logits_override: Optional[torch.Tensor] = None,  # (m_old,) analytic old gate init
        gate_logit_new_override: Optional[float] = None,       # scalar analytic new gate init
        use_align_T: bool = True,
        freeze_B_old: bool = True,
        layer_old: int = 1,
        layer_new: int = 1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.coord_dim_old = coord_dim_old
        self.coord_dim_new = coord_dim_new
        self.m_old = m_old
        self.m_new = m_new
        self.use_align_T = use_align_T
        self.freeze_B_old = freeze_B_old
        self.eps = eps
        self.layer_old = layer_old
        self.layer_new = layer_new

        # --- Static self-consistency scores (not learned) ---
        assert a_old.shape == (m_old,), f"a_old shape {a_old.shape} != ({m_old},)"
        self.register_buffer("a_old", a_old.float().clamp(0, 1))
        # New module scores are always 0 (candidates, not proven yet).
        self.register_buffer("a_new", torch.zeros(m_new))

        # --- Old encoder (frozen columns) ---
        if init_F is not None:
            self.register_buffer("F_old_frozen", init_F.float())
        else:
            self.register_buffer("F_old_frozen",
                                 0.05 * torch.randn(coord_dim_old, m_old))

        # --- Alignment matrix T (m_old x m_old) ---
        # T ≈ I: allows small coordinate rotation without catastrophic forgetting.
        if use_align_T:
            self.align_T = nn.Parameter(torch.eye(m_old))
        else:
            self.register_buffer("align_T", torch.eye(m_old))

        # --- New module encoder (learned) ---
        # Initialize with larger scale so new modules contribute non-trivially
        # from the start. If F_new ≈ 0, gate gradient from data loss ≈ 0,
        # and only L0 sparsity has gradient → gates collapse regardless of w_sp_new.
        if m_new > 0:
            self.F_new = nn.Parameter(
                0.3 * torch.randn(coord_dim_new, m_new))
        else:
            self.register_buffer("F_new", torch.zeros(coord_dim_new, 0))

        # --- Dynamics matrix B ---
        m_total = m_old + m_new
        if init_B is not None:
            if freeze_B_old:
                self.register_buffer("B_old_block", init_B.float())
            else:
                self.B_old_block = nn.Parameter(init_B.float())
        else:
            if freeze_B_old:
                self.register_buffer("B_old_block", torch.eye(m_old) * 0.8)
            else:
                self.B_old_block = nn.Parameter(torch.eye(m_old) * 0.8)

        # New B blocks. With corrected hierarchy:
        #   B_d  (m_new × m_old): old past → new present  (the hierarchy connection)
        #   B_dd (m_new × m_new): new past → new present  (within-layer new dynamics)
        if m_new > 0:
            self.B_d  = nn.Parameter(0.05 * torch.randn(m_new, m_old))  # hierarchy: old→new
            self.B_dd = nn.Parameter(0.05 * torch.randn(m_new, m_new))  # new self-dynamics
        else:
            self.register_buffer("B_d",  torch.zeros(0, m_old))
            self.register_buffer("B_dd", torch.zeros(0, 0))

        # --- Gates (Hard Concrete) ---
        # Old gates: initialized from analytic score or uniform.
        if gate_logits_override is not None:
            old_init = gate_logits_override.float()
        else:
            old_init = torch.full((m_old,), gate_init_logit)
        self.gate_old = HardConcreteGate(m_old, init_logit=0.0)
        self.gate_old.logits = nn.Parameter(old_init)

        # New gates: start strongly positive (logit=+2) so new modules contribute
        # from epoch 1 and F_new receives gradient from the data loss.
        # w_sp_new will prune unnecessary new gates during training.
        # Starting at 0 or negative causes chicken-and-egg: F_new doesn't learn
        # because gate is off, gate doesn't turn on because F_new contributes nothing.
        new_init_logit = (gate_logit_new_override
                          if gate_logit_new_override is not None
                          else 2.0)   # sigmoid(2.0) ≈ 0.88: strongly on by default
        if m_new > 0:
            self.gate_new = HardConcreteGate(m_new, init_logit=new_init_logit)
        else:
            self.gate_new = None

    def assemble_B(self) -> torch.Tensor:
        """
        Full B matrix: (m_old + m_new) x (m_old + m_new).

        Rows = output (module being predicted at t).
        Cols = input (module predicting from t-1).

        Correct hierarchy: old modules are fundamental (lower layer),
        new modules are specific (higher layer). Time-delayed effects
        go fundamental → specific, i.e., old → new only.

          cols:    old_past    new_past
          old_now [B_old_block |   0   ]  old predicted only by old past
          new_now [B_d         | B_dd  ]  new predicted by old past + new past

        B_d   (m_new × m_old): old past → new present  (hierarchy: fundamental drives specific)
        B_dd  (m_new × m_new): new past → new present  (within-layer new dynamics)
        0     (m_old × m_new): new past → old present  FORBIDDEN (specific must not drive fundamental)
        """
        m, n = self.m_old, self.m_new
        if n == 0:
            return self.B_old_block
        # top row: old modules predicted only from old past
        top = torch.cat([self.B_old_block,
                         torch.zeros(m, n, device=self.B_old_block.device)], dim=1)
        # bottom row: new modules predicted from old past (B_d) + new past (B_dd)
        bot = torch.cat([self.B_d, self.B_dd], dim=1)
        return torch.cat([top, bot], dim=0)   # (m+n, m+n)

    def forward(
        self,
        X_proj_old: torch.Tensor,  # (T, 2, coord_dim_old)  new data in old coords
        X_proj_new: torch.Tensor,  # (T, 2, coord_dim_new)  new data in new coords
        tau: float = 1.0,
        hard_gates: bool = False,
    ) -> Dict[str, Any]:
        T = X_proj_old.shape[0]

        # Step 5a: compute old latent with alignment.
        F_old_aligned = self.F_old_frozen @ self.align_T  # (coord_dim_old, m_old)
        Z_old = X_proj_old @ F_old_aligned                # (T, 2, m_old)

        # Step 5b: gate weights = learned gate only.
        # a_old is used ONLY for gate initialization (sign of logit), not as
        # a multiplicative suppressor. Multiplying by a_old (~0.08) permanently
        # kills module contribution regardless of gate value, preventing learning.
        g_old = self.gate_old(tau=tau, hard=hard_gates)   # (m_old,)  in [0,1]
        w_old = g_old                                       # gate is the sole controller

        Z_old_gated = Z_old * w_old.unsqueeze(0).unsqueeze(0)   # (T, 2, m_old)

        # Step 5c: new module latent.
        if self.m_new > 0:
            Z_new = X_proj_new @ self.F_new                # (T, 2, m_new)
            g_new = self.gate_new(tau=tau, hard=hard_gates)
            # New module scores are 0, but we still let g_new work for expansion.
            # The L0 sparsity loss penalizes g_new, so new modules must earn their keep.
            Z_new_gated = Z_new * g_new.unsqueeze(0).unsqueeze(0)
            Z = torch.cat([Z_old_gated, Z_new_gated], dim=-1)  # (T, 2, m_old+m_new)
        else:
            Z = Z_old_gated
            g_new = None

        # Step 5d: dynamics and residual.
        B = self.assemble_B()
        Z_t0 = Z[:, 0, :]   # (T, m_total)
        Z_t1 = Z[:, 1, :]
        E_t1 = Z_t1 - Z_t0 @ B.T   # (T, m_total)

        # Step 6a: data loss — LAD independence (L1 on residuals) + logdet.
        loss_indep = E_t1.abs().mean()
        # Logdet: penalize degenerate encoders. Use only active columns.
        F_active = torch.cat([F_old_aligned * w_old.unsqueeze(0), self.F_new], dim=1) \
            if self.m_new > 0 else F_old_aligned * w_old.unsqueeze(0)
        gram = F_active.T @ F_active + self.eps * torch.eye(
            F_active.shape[1], device=F_active.device)
        loss_logdet = -0.5 * torch.linalg.slogdet(gram)[1]
        loss_data = loss_indep + loss_logdet

        # Step 6b: alignment loss.
        if self.use_align_T:
            loss_align = F.mse_loss(self.align_T,
                                    torch.eye(self.m_old, device=self.align_T.device))
        else:
            loss_align = torch.tensor(0.0, device=Z.device)

        # Step 6c: sparsity on gates (L0 expected active count).
        loss_sp_old = self.gate_old.l0_loss(tau=tau)
        loss_sp_new = self.gate_new.l0_loss(tau=tau) if self.gate_new is not None \
            else torch.tensor(0.0, device=Z.device)

        # Step 6d: new B blocks sparsity.
        loss_B_sp = torch.tensor(0.0, device=Z.device)
        if self.m_new > 0:
            loss_B_sp = self.B_dd.abs().mean() + self.B_d.abs().mean()

        # Step 6e: B regularization (keep new B_old close to init B_old).
        if not self.freeze_B_old:
            loss_B_reg = F.mse_loss(self.B_old_block,
                                    self.B_old_block.detach())
        else:
            loss_B_reg = torch.tensor(0.0, device=Z.device)

        # Gate binary penalty: pushes g toward 0 or 1 (complements Hard Concrete).
        g_all = g_old if g_new is None else torch.cat([g_old, g_new])
        loss_gate_binary = (g_all * (1 - g_all)).mean()

        return {
            "loss_data": loss_data,
            "loss_indep": loss_indep,
            "loss_logdet": loss_logdet,
            "loss_align": loss_align,
            "loss_sp_old": loss_sp_old,
            "loss_sp_new": loss_sp_new,
            "loss_B_sp": loss_B_sp,
            "loss_B_reg": loss_B_reg,
            "loss_gate_binary": loss_gate_binary,
            "g_old": g_old, "g_new": g_new,
            "w_old": w_old,
            "a_old": self.a_old,
            "Z": Z, "E_t1": E_t1, "B": B,
            "F_old_aligned": F_old_aligned,
        }


# =============================================================================
# Training config and loop
# =============================================================================

@dataclass
class TrainConfig:
    name: str
    # Architecture
    use_align_T: bool = True
    freeze_B_old: bool = True
    use_old_gates: bool = True
    use_new_gates: bool = True
    use_analytic_init: bool = True       # use H1/H2/H3 score to init gate logits
    freeze_gates_warmup: bool = True     # freeze gate grads during warmup
    # Loss weights
    w_data: float = 1.0
    w_align: float = 0.01
    w_sp_old: float = 0.01              # L0 sparsity on old gates
    w_sp_new: float = 0.05             # L0 sparsity on new gates (higher = expand less)
    w_B_sp: float = 0.01               # sparsity on new B blocks
    w_B_reg: float = 0.0               # only > 0 when freeze_B_old=False
    w_gate_binary: float = 0.005       # binary penalty
    # Optimization
    lr: float = 0.01
    gate_lr_factor: float = 0.1        # gate logits get slower lr
    epochs: int = 5000
    tau_start: float = 2.0
    tau_end: float = 0.3
    warmup_frac: float = 0.3           # fraction of epochs to freeze gates
    gate_init_logit: float = 0.0       # uniform init logit (if no analytic)
    analytic_gate_C: float = 3.0       # logit magnitude from analytic init
    # Data
    batch_size: int = 0                # 0 = full batch


def train_more(
    model: MoRe,
    X_proj_old: torch.Tensor,   # (T, 2, coord_dim_old)
    X_proj_new: torch.Tensor,   # (T, 2, coord_dim_new)
    cfg: TrainConfig,
    device: torch.device,
    verbose_every: int = 0,
) -> Dict[str, Any]:
    model = model.to(device)
    X_proj_old = X_proj_old.to(device)
    X_proj_new = X_proj_new.to(device)

    # Separate gate and non-gate parameters for different learning rates.
    gate_params, non_gate_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "gate" in n:
            gate_params.append(p)
        else:
            non_gate_params.append(p)

    param_groups = [{"params": non_gate_params, "lr": cfg.lr}]
    if gate_params:
        param_groups.append({"params": gate_params, "lr": cfg.lr * cfg.gate_lr_factor})
    opt = torch.optim.Adam(param_groups)

    t0 = time.perf_counter()
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        frac = epoch / max(cfg.epochs - 1, 1)
        tau = cfg.tau_start * (cfg.tau_end / cfg.tau_start) ** frac

        out = model(X_proj_old, X_proj_new, tau=tau, hard_gates=False)

        loss = (
            cfg.w_data        * out["loss_data"]
            + cfg.w_align     * out["loss_align"]
            + cfg.w_sp_old    * out["loss_sp_old"]
            + cfg.w_sp_new    * out["loss_sp_new"]
            + cfg.w_B_sp      * out["loss_B_sp"]
            + cfg.w_B_reg     * out["loss_B_reg"]
            + cfg.w_gate_binary * out["loss_gate_binary"]
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # Freeze gate gradients during warmup (only for analytic init variant).
        if cfg.freeze_gates_warmup and frac < cfg.warmup_frac and gate_params:
            for p in gate_params:
                if p.grad is not None:
                    p.grad.zero_()

        opt.step()

        if verbose_every and (epoch + 1) % verbose_every == 0:
            g = to_np(out["g_old"])
            gn = to_np(out["g_new"]) if out["g_new"] is not None else None
            print(
                f"[{cfg.name}] ep {epoch+1}/{cfg.epochs} "
                f"loss={loss.item():.4f} indep={out['loss_indep'].item():.4f} "
                f"g_old={g.round(3).tolist()} "
                + (f"g_new={gn.round(3).tolist()}" if gn is not None else ""),
                flush=True,
            )
        if (epoch + 1) % max(cfg.epochs // 10, 1) == 0:
            history.append({
                "epoch": epoch + 1,
                "loss": loss.item(),
                "loss_indep": out["loss_indep"].item(),
                "g_old": to_np(out["g_old"]).tolist(),
            })

    # Final eval with hard gates.
    model.eval()
    with torch.no_grad():
        out_hard = model(X_proj_old, X_proj_new, tau=cfg.tau_end, hard_gates=True)
        out_soft = model(X_proj_old, X_proj_new, tau=cfg.tau_end, hard_gates=False)

    return {
        "out_hard": {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                     for k, v in out_hard.items()},
        "out_soft": {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                     for k, v in out_soft.items()},
        "train_time_sec": time.perf_counter() - t0,
        "n_params": count_params(model),
        "history": history,
    }


# =============================================================================
# Evaluation utilities
# =============================================================================

def estimate_Z(
    model: MoRe, X_proj_old: torch.Tensor, X_proj_new: torch.Tensor,
    device: torch.device, hard: bool = True,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(X_proj_old.to(device), X_proj_new.to(device),
                    tau=0.1, hard_gates=hard)
    return out["Z"].cpu()


def corr_abs(Z_hat: torch.Tensor, Z_true: torch.Tensor) -> torch.Tensor:
    A = Z_hat.reshape(-1, Z_hat.shape[-1]).float()
    B = Z_true.reshape(-1, Z_true.shape[-1]).float()
    A = (A - A.mean(0)) / (A.std(0) + 1e-8)
    B = (B - B.mean(0)) / (B.std(0) + 1e-8)
    return ((A.T @ B) / max(A.shape[0] - 1, 1)).abs()


def best_mcc(
    Z_hat: torch.Tensor, Z_true: torch.Tensor, true_indices: Sequence[int],
) -> Tuple[float, List]:
    if len(true_indices) == 0:
        return 0.0, []
    Zt = Z_true[:, :, list(true_indices)]
    C = corr_abs(Z_hat, Zt)
    n_hat, n_true = C.shape
    best_s, best_p = -1.0, []
    for perm in permutations(range(n_hat), min(n_hat, n_true)):
        vals = [float(C[perm[j], j]) for j in range(len(perm))]
        s = float(np.mean(vals))
        if s > best_s:
            best_s = s
            best_p = [(int(perm[j]), int(true_indices[j]), vals[j])
                      for j in range(len(perm))]
    return best_s, best_p


def gate_accuracy(
    g_old: Optional[torch.Tensor],
    old_map: Dict[int, int],
    target_active: Sequence[int],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    if g_old is None:
        return {"gate_acc": float("nan"), "tp": 0, "tn": 0, "fp": 0, "fn": 0,
                "n_active": float("nan")}
    g = g_old.detach().cpu()
    target_set = set(int(x) for x in target_active)
    y_true = [1 if old_map.get(i, -1) in target_set else 0 for i in range(len(g))]
    y_pred = [1 if float(g[i]) > threshold else 0 for i in range(len(g))]
    acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else float("nan")
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    return {"gate_acc": acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "n_active": int((g > threshold).sum())}