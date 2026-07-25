#!/usr/bin/env python3
"""Leakage-free gate experiment for the larger C11--C17 benchmark.

This script is intentionally self-contained within ``synthetic_gate_inspection``.
It leaves the published C1--C7 generator and runners untouched.

The data have three causal layers of five independent, non-Gaussian sources:
L1 -> L2 -> L3.  The first dataset always contains all five L1 sources.  The
seven second-dataset settings are the scaled counterparts of C1--C7:

  C11: retain all L1                         (5 retained, 0 added)
  C12: retain L1 and add L2,L3              (5 retained, 10 added)
  C13: discard L1 and use L2,L3             (0 retained, 10 added)
  C14: retain L1[0:2] and add L2,L3          (2 retained, 10 added)
  C15: retain L1[2:5] and add L2,L3          (3 retained, 10 added)
  C16: retain only L1[0:2]                   (2 retained, 0 added)
  C17: retain only L1[2:5]                   (3 retained, 0 added)

Structural selection receives only old and target observations plus the
frozen old-model output.  Ground-truth factors and scenario masks are used
only after adaptation for evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "synthetic_adaptation"
sys.path.insert(0, str(SOURCE))

from more_baselines import train_old_model  # noqa: E402
from more_core import (  # noqa: E402
    TrainConfig, compute_self_consistency, estimate_Z, expand_basis,
    fit_pca_basis, set_seed, train_more,
)
from rank_aware_gate import decision_metadata, install_fixed_gate, leakage_free_ica_gate  # noqa: E402


N_PER_LAYER = 5
N_FACTORS = 3 * N_PER_LAYER
OLD_FACTORS = list(range(N_PER_LAYER))

# Name, active global latent indices in dataset 2.  This is deliberately
# passed only to evaluation code, never to ``leakage_free_ica_gate``.
SCENARIOS: Tuple[Tuple[str, List[int]], ...] = (
    ("C11_same_L1", list(range(0, 5))),
    ("C12_L1_L2_L3", list(range(0, 15))),
    ("C13_L2_L3", list(range(5, 15))),
    ("C14_L1_01_L2_L3", [0, 1] + list(range(5, 15))),
    ("C15_L1_234_L2_L3", [2, 3, 4] + list(range(5, 15))),
    ("C16_L1_01_only", [0, 1]),
    ("C17_L1_234_only", [2, 3, 4]),
)


class FixedHardGate(nn.Module):
    """Minimal HardConcrete-compatible interface for a fixed gate decision."""

    def __init__(self, n: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n))

    def forward(self, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        values = torch.sigmoid(self.logits)
        return (values > 0.5).float() if hard else values

    def l0_loss(self, tau: float = 1.0) -> torch.Tensor:
        return torch.sigmoid(self.logits).sum()


class DeepMoRe(nn.Module):
    """Fixed-gate MoRe variant with unequal old/new coordinate widths.

    The original ``MoRe`` combines old and new encoder matrices before its
    log-determinant regularizer, valid only when both coordinate blocks have
    the same width. Here the blocks occupy orthogonal PCA coordinates, so the
    appropriate Gram matrix is block diagonal. Everything relevant to the
    structural experiment remains the same: frozen old encoder/dynamics,
    learned new encoder/dynamics, and the old -> new hierarchy.
    """

    def __init__(self, old_coord_dim: int, new_coord_dim: int, m_old: int,
                 m_new: int, F_old: torch.Tensor, B_old: torch.Tensor):
        super().__init__()
        self.m_old, self.m_new = m_old, m_new
        self.register_buffer("F_old_frozen", F_old.float())
        self.register_buffer("B_old_block", B_old.float())
        self.gate_old = FixedHardGate(m_old)
        self.gate_new = FixedHardGate(m_new) if m_new else None
        if m_new:
            self.F_new = nn.Parameter(0.3 * torch.randn(new_coord_dim, m_new))
            self.B_d = nn.Parameter(0.05 * torch.randn(m_new, m_old))
            self.B_dd = nn.Parameter(0.05 * torch.randn(m_new, m_new))
        else:
            self.register_buffer("F_new", torch.zeros(new_coord_dim, 0))
            self.register_buffer("B_d", torch.zeros(0, m_old))
            self.register_buffer("B_dd", torch.zeros(0, 0))

    def assemble_B(self) -> torch.Tensor:
        if not self.m_new:
            return self.B_old_block
        top = torch.cat([self.B_old_block,
                         torch.zeros(self.m_old, self.m_new, device=self.B_old_block.device)], dim=1)
        return torch.cat([top, torch.cat([self.B_d, self.B_dd], dim=1)], dim=0)

    def forward(self, X_proj_old: torch.Tensor, X_proj_new: torch.Tensor,
                tau: float = 1.0, hard_gates: bool = False) -> Dict[str, Any]:
        g_old = self.gate_old(tau=tau, hard=hard_gates)
        Z_old = (X_proj_old @ self.F_old_frozen) * g_old.view(1, 1, -1)
        if self.m_new:
            g_new = self.gate_new(tau=tau, hard=hard_gates)
            Z_new = (X_proj_new @ self.F_new) * g_new.view(1, 1, -1)
            Z = torch.cat([Z_old, Z_new], dim=-1)
        else:
            g_new, Z = None, Z_old
        B = self.assemble_B()
        residual = Z[:, 1] - Z[:, 0] @ B.T
        eps = 1e-6
        old_gram = (self.F_old_frozen * g_old.view(1, -1)).T @ (self.F_old_frozen * g_old.view(1, -1))
        logdet = torch.linalg.slogdet(old_gram + eps * torch.eye(self.m_old, device=Z.device))[1]
        if self.m_new:
            new_gram = self.F_new.T @ self.F_new
            logdet = logdet + torch.linalg.slogdet(new_gram + eps * torch.eye(self.m_new, device=Z.device))[1]
        zero = torch.zeros((), device=Z.device)
        b_sp = (self.B_d.abs().mean() + self.B_dd.abs().mean()) if self.m_new else zero
        all_gates = torch.cat([g_old, g_new]) if g_new is not None else g_old
        return {
            "loss_data": residual.abs().mean() - 0.5 * logdet,
            "loss_indep": residual.abs().mean(), "loss_logdet": -0.5 * logdet,
            "loss_align": zero, "loss_sp_old": self.gate_old.l0_loss(tau),
            "loss_sp_new": self.gate_new.l0_loss(tau) if self.gate_new else zero,
            "loss_B_sp": b_sp, "loss_B_reg": zero,
            "loss_gate_binary": (all_gates * (1 - all_gates)).mean(),
            "g_old": g_old, "g_new": g_new, "w_old": g_old,
            "Z": Z, "E_t1": residual, "B": B,
        }


@dataclass
class DeepSyntheticData:
    """Observations and evaluation-only latent factors for one data seed."""

    Z_full: torch.Tensor
    X_full: torch.Tensor
    mixing_A: torch.Tensor
    scenarios: Dict[str, Dict[str, torch.Tensor]]


def generate_three_layer_data(T: int, obs_dim: int, latent_scale: float,
                              data_seed: int) -> DeepSyntheticData:
    """Generate a linear three-layer, five-components-per-layer data set.

    At t=0 all 15 sources are mutually independent Laplace variables.  At
    t=1, L1 has within-layer lagged links; L2 depends on L1 and itself; and L3
    depends on L1, L2 and itself.  Thus the data are identifiable by ICA at
    t=0, while the t=1 transition has the requested three-layer structure.
    """
    set_seed(data_seed)
    lap = torch.distributions.Laplace(0.0, latent_scale)
    E = lap.sample((T, 2, N_FACTORS))
    Z = E.clone()
    z0 = E[:, 0, :]

    # Fixed sparse, directed within-layer effects; coefficients are below one
    # to keep the generated process well-conditioned.
    W11 = torch.zeros(5, 5)
    W11[1, 0], W11[2, 1], W11[3, 2], W11[4, 3] = 0.18, 0.16, 0.14, 0.12
    W22 = torch.zeros(5, 5)
    W22[1, 0], W22[2, 1], W22[3, 2], W22[4, 3] = 0.14, 0.12, 0.10, 0.08
    W33 = torch.zeros(5, 5)
    W33[1, 0], W33[2, 1], W33[3, 2], W33[4, 3] = 0.12, 0.10, 0.08, 0.06
    W21 = torch.tensor([
        [0.50, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.45, 0.00, 0.00, 0.00],
        [0.20, 0.00, 0.40, 0.00, 0.00],
        [0.00, 0.15, 0.00, 0.35, 0.00],
        [0.10, 0.00, 0.10, 0.00, 0.30],
    ])
    W31 = torch.tensor([
        [0.25, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.22, 0.00, 0.00, 0.00],
        [0.10, 0.00, 0.20, 0.00, 0.00],
        [0.00, 0.08, 0.00, 0.18, 0.00],
        [0.06, 0.00, 0.06, 0.00, 0.16],
    ])
    W32 = torch.tensor([
        [0.45, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.40, 0.00, 0.00, 0.00],
        [0.15, 0.00, 0.35, 0.00, 0.00],
        [0.00, 0.12, 0.00, 0.30, 0.00],
        [0.08, 0.00, 0.08, 0.00, 0.28],
    ])

    l1, l2, l3 = z0[:, :5], z0[:, 5:10], z0[:, 10:15]
    Z[:, 1, :5] = E[:, 1, :5] + 0.65 * l1 + l1 @ W11.T
    Z[:, 1, 5:10] = E[:, 1, 5:10] + 0.62 * l2 + l2 @ W22.T + l1 @ W21.T
    Z[:, 1, 10:15] = (E[:, 1, 10:15] + 0.60 * l3 + l3 @ W33.T
                        + l1 @ W31.T + l2 @ W32.T)

    # Rectangular full-row-rank mixing makes every active-set rank observable.
    mixing_A = torch.randn(N_FACTORS, obs_dim)
    mixing_A = mixing_A / mixing_A.norm(dim=1, keepdim=True).clamp_min(1e-8)
    X = Z @ mixing_A

    def masked(active: Sequence[int]) -> Dict[str, torch.Tensor]:
        mask = torch.zeros(N_FACTORS, dtype=torch.bool)
        mask[list(active)] = True
        Zm = Z.clone()
        Zm[:, :, ~mask] = 0.0
        return {"Z": Zm, "X": Zm @ mixing_A, "active": torch.tensor(active)}

    scenarios = {"D1_L1": masked(OLD_FACTORS)}
    scenarios.update({name: masked(active) for name, active in SCENARIOS})
    return DeepSyntheticData(Z, X, mixing_A, scenarios)


def hungarian_mcc(Z_hat: torch.Tensor, Z_true: torch.Tensor,
                  true_indices: Sequence[int]) -> Tuple[float, List[Tuple[int, int, float]]]:
    """Original top-k absolute-correlation MCC, scaled beyond four factors.

    ``linear_sum_assignment`` exactly maximizes the same one-to-one mean MCC
    objective as ``more_core.best_mcc`` when there are at least as many model
    modules as active ground-truth dimensions.  It avoids factorial enumeration
    for the 15-dimensional C12 benchmark.
    """
    if not true_indices:
        return 0.0, []
    A = Z_hat.reshape(-1, Z_hat.shape[-1]).float()
    B = Z_true[:, :, list(true_indices)].reshape(-1, len(true_indices)).float()
    A = (A - A.mean(0)) / A.std(0).clamp_min(1e-8)
    B = (B - B.mean(0)) / B.std(0).clamp_min(1e-8)
    C = ((A.T @ B) / max(A.shape[0] - 1, 1)).abs().cpu().numpy()
    rows, cols = linear_sum_assignment(-C)
    if len(cols) != len(true_indices):
        raise RuntimeError("not enough estimated modules for the requested MCC")
    pairs = [(int(r), int(true_indices[c]), float(C[r, c])) for r, c in zip(rows, cols)]
    return float(np.mean([p[2] for p in pairs])), pairs


def gate_stats(g_old: torch.Tensor, old_mapping: Dict[int, int],
               target_active: Sequence[int]) -> Dict[str, Any]:
    target = set(map(int, target_active))
    truth = np.array([old_mapping.get(i, -1) in target for i in range(len(g_old))])
    pred = g_old.detach().cpu().numpy() > 0.5
    return {
        "gate_acc": float((truth == pred).mean()),
        "tp": int(np.sum(truth & pred)), "tn": int(np.sum(~truth & ~pred)),
        "fp": int(np.sum(~truth & pred)), "fn": int(np.sum(truth & ~pred)),
    }


@torch.no_grad()
def estimate_all_modules(model: nn.Module, X_old: torch.Tensor, X_new: torch.Tensor,
                         device: torch.device) -> torch.Tensor:
    """Retention evaluation: enable all learned modules, then take top-k MCC."""
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


def any_trainable(model: torch.nn.Module) -> bool:
    return any(p.requires_grad for p in model.parameters())


def initialize_new_encoder_ica(model: DeepMoRe, X_target_new: torch.Tensor,
                               random_state: int) -> None:
    """Initialize the new branch by ICA on unlabeled residual target coordinates."""
    if not model.m_new:
        return
    from sklearn.decomposition import FastICA
    samples = X_target_new[:, 0, :].detach().cpu().numpy()
    ica = FastICA(n_components=model.m_new, whiten="unit-variance",
                  random_state=random_state, max_iter=2_000, tol=1e-5).fit(samples)
    # sklearn components_ maps row samples to sources: sources = X @ components_.T.
    model.F_new.data.copy_(torch.from_numpy(ica.components_.T).to(model.F_new))


def canonicalize_old_encoder_ica(X_old_pca: torch.Tensor, random_state: int) -> Dict[str, torch.Tensor]:
    """Resolve old linear-encoder rotation ambiguity with unlabeled old-data ICA.

    A linear old encoder can represent any rotation of its five sources.  A
    component-wise gate needs a stable component coordinate system, so we use
    the same identifiable non-Gaussian criterion already used for target
    initialization.  Dynamics are then refit by ordinary least squares in
    those canonical coordinates.  No target observation or ground truth is
    involved.
    """
    from sklearn.decomposition import FastICA
    samples = X_old_pca[:, 0, :].detach().cpu().numpy()
    ica = FastICA(n_components=N_PER_LAYER, whiten="unit-variance",
                  random_state=random_state, max_iter=2_000, tol=1e-5).fit(samples)
    F_old = torch.from_numpy(ica.components_.T).float()
    Z_old = X_old_pca @ F_old
    B_old = torch.linalg.lstsq(Z_old[:, 0, :], Z_old[:, 1, :]).solution.T.contiguous()
    return {"F": F_old, "B": B_old, "Z": Z_old}


def one_run(data: DeepSyntheticData, case_name: str, target_active: Sequence[int],
            data_seed: int, train_seed: int, args: argparse.Namespace,
            device: torch.device) -> Dict[str, Any]:
    """Fit one training replicate. Ground truth enters only after `estimate_Z`."""
    # Deliberately reseed after data generation: train_seed controls all model
    # initialization/optimisation variation and cannot affect the dataset.
    set_seed(train_seed)
    old, target = data.scenarios["D1_L1"], data.scenarios[case_name]
    basis_old = fit_pca_basis(old["X"], N_PER_LAYER)
    X_old_pca = basis_old.project(old["X"])
    # Retain the original old-model fitting stage, then canonically express its
    # frozen representation in ICA coordinates so a module gate is meaningful.
    _, trained_old_info = train_old_model(X_old_pca, N_PER_LAYER, args.lr,
                                          args.old_epochs, device)
    old_info = (canonicalize_old_encoder_ica(X_old_pca, 30_000 + data_seed)
                if args.old_encoder == "ica_canonical" else trained_old_info)
    F_old, B_old = old_info["F"], old_info["B"]

    # Evaluation-only mapping for gate accuracy, constructed after old training.
    old_reference_mcc, old_pairs = hungarian_mcc(old_info["Z"], old["Z"], OLD_FACTORS)
    old_mapping = {module: truth for module, truth, _ in old_pairs}

    # --------- Leakage-free structural decision begins ---------
    # Only X_old, X_target and frozen old encoder outputs are arguments here.
    decision = leakage_free_ica_gate(
        old["X"], target["X"], old_info["Z"], args.rank_rtol,
        args.n_bootstrap, args.stability_margin, random_state=10_000 + data_seed,
        threshold_override=args.persistence_threshold,
    )
    # --------- No latent labels / target mask were supplied above. ---------
    n_new = int(decision.g_new.numel())
    _, V_new = expand_basis(target["X"], basis_old, n_new)
    X_target_old = basis_old.project(target["X"])
    X_target_new = ((target["X"] - basis_old.mean) @ V_new if n_new else
                    torch.zeros(target["X"].shape[0], 2, 0))
    a_old, _ = compute_self_consistency(X_target_old.to(device), F_old.to(device), B_old.to(device))

    model = DeepMoRe(N_PER_LAYER, n_new, N_PER_LAYER, n_new, F_old, B_old)
    install_fixed_gate(model, decision)
    if args.new_init == "ica":
        initialize_new_encoder_ica(model, X_target_new, 20_000 + data_seed)
    cfg = TrainConfig(
        name="C11_C17_leakfree_fixed_gate", use_align_T=False, freeze_B_old=True,
        use_old_gates=True, use_new_gates=True, epochs=args.epochs, lr=args.lr,
        tau_start=args.tau_start, tau_end=args.tau_end, warmup_frac=0.0,
        w_sp_old=0.0, w_sp_new=0.0, w_B_sp=args.w_B_sp,
        w_gate_binary=0.0, gate_lr_factor=0.0,
    )
    if any_trainable(model):
        info = train_more(model, X_target_old, X_target_new, cfg, device)
        train_time = float(info["train_time_sec"])
    else:
        train_time = 0.0

    Z_target = estimate_Z(model, X_target_old, X_target_new, device)
    plasticity, _ = hungarian_mcc(Z_target, target["Z"], target_active)
    new_truth = [i for i in target_active if i >= N_PER_LAYER]
    new_mcc, _ = hungarian_mcc(Z_target[:, :, N_PER_LAYER:], target["Z"], new_truth) if new_truth else (float("nan"), [])

    # Primary retention metric: every module remains available.  The old
    # encoder/basis/dynamics are frozen; top-k MCC evaluates the old factors.
    X_old_new = ((old["X"] - basis_old.mean) @ V_new if n_new else
                 torch.zeros(old["X"].shape[0], 2, 0))
    Z_old_all = estimate_all_modules(model, X_old_pca, X_old_new, device)
    post_adaptation_mcc, _ = hungarian_mcc(Z_old_all, old["Z"], OLD_FACTORS)
    # Since all frozen old paths are available, post-MCC can occasionally be
    # higher than the old reference (an added module happens to align better).
    # That is positive transfer, not negative forgetting.
    forgetting = max(0.0, old_reference_mcc - post_adaptation_mcc)

    # This diagnostic answers a different question: availability through the
    # target-selected branch. It is not the model-retention forgetting metric.
    Z_old_selected = estimate_Z(model, X_old_pca, X_old_new, device)
    selected_branch_mcc, _ = hungarian_mcc(Z_old_selected, old["Z"], OLD_FACTORS)
    stats = gate_stats(decision.g_old, old_mapping, target_active)
    return {
        "case": case_name, "data_seed": data_seed, "train_seed": train_seed,
        "plasticity_mcc": plasticity, "new_factor_mcc": new_mcc,
        "old_reference_MCC": old_reference_mcc,
        "post_adaptation_top_k_MCC": post_adaptation_mcc,
        "forgetting": forgetting,
        "selected_branch_old_MCC_aux": selected_branch_mcc,
        "selected_branch_forgetting_aux": max(0.0, old_reference_mcc - selected_branch_mcc),
        "gate_acc": stats["gate_acc"], **stats,
        "new_init": args.new_init,
        "old_encoder": args.old_encoder,
        "train_time_sec": train_time,
        **decision_metadata(decision),
    }


def write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, default=HERE / "results" / "c11_c17")
    p.add_argument("--data_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--train_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--cases", nargs="+", default=[name for name, _ in SCENARIOS])
    p.add_argument("--T", type=int, default=5_000)
    p.add_argument("--obs_dim", type=int, default=20)
    p.add_argument("--latent_scale", type=float, default=0.1)
    p.add_argument("--old_epochs", type=int, default=300)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--tau_start", type=float, default=2.0)
    p.add_argument("--tau_end", type=float, default=0.3)
    p.add_argument("--w_B_sp", type=float, default=0.01)
    p.add_argument("--rank_rtol", type=float, default=1e-3)
    p.add_argument("--n_bootstrap", type=int, default=4)
    p.add_argument("--stability_margin", type=float, default=0.02)
    p.add_argument("--persistence_threshold", type=float, default=0.90,
                   help="fixed label-free ICA cosine cutoff; use -1 for bootstrap-calibrated cutoff")
    p.add_argument("--new_init", choices=["ica", "random"], default="ica",
                   help="unlabeled initialization for newly added target modules")
    p.add_argument("--old_encoder", choices=["ica_canonical", "trained"], default="ica_canonical",
                   help="canonicalize the trained old representation before component-wise gating")
    p.add_argument("--resume", action="store_true",
                   help="retain completed (case, data seed, train seed) rows and run only missing fits")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    valid = {name for name, _ in SCENARIOS}
    unknown = set(args.cases) - valid
    if unknown:
        p.error(f"unknown cases: {sorted(unknown)}")
    if args.obs_dim < N_FACTORS:
        p.error("obs_dim must be at least 15 so full C12 rank is observable")
    if args.persistence_threshold < 0:
        args.persistence_threshold = None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args) | {"out_dir": str(args.out_dir)}, indent=2, default=str))

    by_name = dict(SCENARIOS)
    metrics_path = args.out_dir / "metrics.csv"
    rows: List[Dict[str, Any]] = read_csv(metrics_path) if args.resume else []
    completed = {(r["case"], int(r["data_seed"]), int(r["train_seed"])) for r in rows}
    wall = time.perf_counter()
    for data_seed in args.data_seeds:
        data = generate_three_layer_data(args.T, args.obs_dim, args.latent_scale, data_seed)
        torch.save({"Z_full": data.Z_full, "X_full": data.X_full, "mixing_A": data.mixing_A,
                    "scenarios": data.scenarios}, args.out_dir / f"data_seed{data_seed}.pt")
        for train_seed in args.train_seeds:
            for case in args.cases:
                key = (case, data_seed, train_seed)
                if key in completed:
                    print(f"skip completed {case} data={data_seed} train={train_seed}", flush=True)
                    continue
                row = one_run(data, case, by_name[case], data_seed, train_seed, args, torch.device(args.device))
                rows.append(row)
                print(f"{case} data={data_seed} train={train_seed}: gate={row['gate_acc']:.3f} "
                      f"plasticity={row['plasticity_mcc']:.3f} forgetting={row['forgetting']:.6f}", flush=True)
                completed.add(key)
                write_csv(rows, metrics_path)
    write_csv(rows, metrics_path)
    (args.out_dir / "run_time_seconds.txt").write_text(f"{time.perf_counter() - wall:.3f}\n")
    print(f"completed {len(rows)} runs in {time.perf_counter() - wall:.1f}s; run summarize_c11_c17.py next", flush=True)


if __name__ == "__main__":
    main()
