import copy
import csv
import itertools
import json
import math
import os
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.temporal_cfg import PRESET_CONFIGS


BASELINE_PRESETS = (
    "linear_gaussian",
    "linear_laplace",
    "linear_uniform",
    "lg_nonlinear_mixing",
    "ll_nonlinear_mixing",
    "lu_nonlinear_mixing",
    "g_nonlinear",
    "l_nonlinear",
    "u_nonlinear",
)

TARGET_LAYER1_PRESETS = {
    0.05: "three_layer_l_nonlinear_layer1_var_005",
    0.10: "three_layer_l_nonlinear_layer1_var_010",
    0.20: "three_layer_l_nonlinear_layer1_var_020",
    0.50: "three_layer_l_nonlinear_layer1_var_050",
}

TARGET_LAYER2_PRESETS = {
    0.05: "three_layer_l_nonlinear_layer2_var_005",
    0.10: "three_layer_l_nonlinear_layer2_var_010",
    0.20: "three_layer_l_nonlinear_layer2_var_020",
    0.50: "three_layer_l_nonlinear_layer2_var_050",
}


@dataclass(frozen=True)
class DatasetSpec:
    preset: str
    command: str
    output_path: Path
    n_layers: int
    n_vars_per_layer: int


@dataclass
class TrainConfig:
    lr_q: float = 1e-3
    lr_f: float = 3e-4
    lr_pred: float = 3e-4
    lambda_jepa: float = 0.1
    lambda_recon: float = 0.1
    lambda_cmi: float = 1e-2
    n_epochs: int = 1000
    n_q_epochs: int = 20
    n_f_epochs: int = 3
    hid_dim: int = 64
    hid_layer: int = 2
    linear: bool = False
    ema_m: float = 0.95
    use_jepa_warmup: bool = True
    warmup_epochs: int = 50
    seed: int = 0
    snapshot_epochs: Tuple[int, ...] = (1, 10, 25, 50, 100, 200, 500)
    scatter_points: int = 4000

    def to_json_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["snapshot_epochs"] = list(self.snapshot_epochs)
        return payload


@dataclass
class DatasetBundle:
    name: str
    npz_path: Path
    raw_xt: np.ndarray
    raw_zt: np.ndarray
    xt0: torch.Tensor
    xt1: torch.Tensor
    xt0_norm: torch.Tensor
    xt1_norm: torch.Tensor
    zt0: torch.Tensor
    zt1: torch.Tensor
    x_mean: torch.Tensor
    x_std: torch.Tensor
    n_layers: int
    n_vars_per_layer: int
    latent_names: List[str]

    @property
    def first_layer_indices(self) -> List[int]:
        return list(range(self.n_vars_per_layer))

    @property
    def non_first_layer_indices(self) -> List[int]:
        return list(range(self.n_vars_per_layer, self.raw_zt.shape[-1]))


@dataclass
class TrainingState:
    f1: nn.Module
    f1_teacher: nn.Module
    q_pred: nn.Module
    g1: nn.Module
    q_z1t1_xt0: nn.Module
    q_z1t1_z1t0: nn.Module
    optim_q: torch.optim.Optimizer
    optim_f: torch.optim.Optimizer
    optim_pred: torch.optim.Optimizer
    train_cfg: TrainConfig
    device: str
    pred_tol: Optional[float] = None
    global_epoch: int = 0


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 256, depth: int = 2, act: str = "gelu"):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, width)
        self.out = nn.Linear(width, out_dim)

        if act == "gelu":
            self.act = nn.GELU
        elif act == "tanh":
            self.act = nn.Tanh
        elif act == "relu":
            self.act = nn.ReLU
        else:
            raise ValueError(f"Unknown activation {act}")

        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, width), self.act()) for _ in range(depth)])
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = F.dropout(h, 0.2, training=self.training)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


class GaussianCondDensityEstimator(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hid_dim: int,
        hid_layers: int,
        logvar_min: float = -5.0,
        logvar_max: float = 2.0,
        learn_logvar: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mu = MLP(in_dim=in_dim, out_dim=out_dim, width=hid_dim, depth=hid_layers)
        self.logvar = MLP(in_dim=in_dim, out_dim=out_dim, width=hid_dim, depth=hid_layers)
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        if not learn_logvar:
            self.logvar_min = 0.0
            self.logvar_max = 0.0

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        mu = self.mu(cond)
        logvar = self.logvar(cond)
        logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        var = torch.exp(logvar)
        centered = x - mu.reshape(-1, self.out_dim)
        return (
            -0.5 * self.out_dim * torch.tensor(2 * torch.pi, device=cond.device, dtype=cond.dtype).log()
            - 0.5 * torch.log(var).sum(dim=-1)
            - 0.5 * (centered ** 2 * (1.0 / var)).sum(dim=-1)
        )


class EncoderF1(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hid_dim: int, hid_layers: int, linear: bool = False, act: str = "gelu"):
        super().__init__()
        if linear:
            self.encoder_f1 = nn.Linear(in_dim, out_dim)
        else:
            self.encoder_f1 = MLP(in_dim=in_dim, out_dim=out_dim, width=hid_dim, depth=hid_layers, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_f1(x)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_dict_to_numpy(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy() for key, value in state_dict.items()}


def save_model_bundle_npz(save_path: Path, models: Dict[str, nn.Module], metadata: Dict[str, object]) -> None:
    save_dict: Dict[str, np.ndarray] = {}
    for model_name, model in models.items():
        for key, value in state_dict_to_numpy(model.state_dict()).items():
            save_dict[f"{model_name}.{key}"] = value
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            save_dict[key] = value.detach().cpu().numpy()
        else:
            save_dict[key] = np.asarray(value)
    ensure_dir(save_path.parent)
    np.savez(save_path, **save_dict)


def parse_generator_commands(script_path: Path) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    line_re = re.compile(r"--preset\s+(\S+).+?--save\s+(\S+)")
    for raw_line in script_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = line_re.search(line)
        if not match:
            continue
        preset = match.group(1)
        save_arg = match.group(2)
        cfg = PRESET_CONFIGS[preset]
        output_path = (script_path.parent / save_arg).resolve()
        if "--multi" in line:
            output_path = output_path.with_name(f"multi_{output_path.name}")
        specs.append(
            DatasetSpec(
                preset=preset,
                command=line,
                output_path=output_path,
                n_layers=cfg.n_layers,
                n_vars_per_layer=cfg.n_vars_per_layer,
            )
        )
    return specs


def run_generation(specs: Sequence[DatasetSpec], script_dir: Path, force: bool = False) -> None:
    for spec in specs:
        if spec.output_path.exists() and not force:
            continue
        ensure_dir(spec.output_path.parent)
        subprocess.run(spec.command, cwd=script_dir, shell=True, check=True)


def load_dataset_bundle(
    npz_path: Path,
    dataset_name: str,
    device: str,
    n_layers: int,
    n_vars_per_layer: int,
) -> DatasetBundle:
    raw = np.load(npz_path)
    raw_xt = np.asarray(raw["xt"], dtype=np.float32)
    raw_zt = np.asarray(raw["zt"], dtype=np.float32)
    xt0 = torch.tensor(raw_xt[:, 0, :], device=device, dtype=torch.float32)
    xt1 = torch.tensor(raw_xt[:, 1, :], device=device, dtype=torch.float32)
    zt0 = torch.tensor(raw_zt[:, 0, :], device=device, dtype=torch.float32)
    zt1 = torch.tensor(raw_zt[:, 1, :], device=device, dtype=torch.float32)
    x_mean = xt0.mean(dim=0, keepdim=True)
    x_std = xt0.std(dim=0, keepdim=True) + 1e-8
    xt0_norm = (xt0 - x_mean) / x_std
    xt1_norm = (xt1 - x_mean) / x_std
    latent_names = []
    for layer_idx in range(n_layers):
        for var_idx in range(n_vars_per_layer):
            latent_names.append(f"L{layer_idx + 1}V{var_idx + 1}")
    return DatasetBundle(
        name=dataset_name,
        npz_path=npz_path,
        raw_xt=raw_xt,
        raw_zt=raw_zt,
        xt0=xt0,
        xt1=xt1,
        xt0_norm=xt0_norm,
        xt1_norm=xt1_norm,
        zt0=zt0,
        zt1=zt1,
        x_mean=x_mean,
        x_std=x_std,
        n_layers=n_layers,
        n_vars_per_layer=n_vars_per_layer,
        latent_names=latent_names,
    )


def build_optimizers(
    f1: nn.Module,
    g1: nn.Module,
    q_pred: nn.Module,
    q_z1t1_xt0: nn.Module,
    q_z1t1_z1t0: nn.Module,
    train_cfg: TrainConfig,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer]:
    optim_q = torch.optim.Adam(
        itertools.chain(q_z1t1_xt0.parameters(), q_z1t1_z1t0.parameters()),
        lr=train_cfg.lr_q,
    )
    optim_f = torch.optim.Adam(
        itertools.chain(f1.parameters(), g1.parameters()),
        lr=train_cfg.lr_f,
    )
    optim_pred = torch.optim.Adam(q_pred.parameters(), lr=train_cfg.lr_pred)
    return optim_q, optim_f, optim_pred


def initialize_training_state(dataset: DatasetBundle, train_cfg: TrainConfig, device: str) -> TrainingState:
    if train_cfg.lr_f > train_cfg.lr_q or train_cfg.lr_pred > train_cfg.lr_q:
        raise ValueError("lr_f and lr_pred must not exceed lr_q.")
    if train_cfg.n_q_epochs <= train_cfg.n_f_epochs:
        raise ValueError("n_q_epochs should be greater than n_f_epochs.")

    set_seeds(train_cfg.seed)
    d = dataset.raw_xt.shape[-1]
    f1 = EncoderF1(
        in_dim=d,
        out_dim=1,
        hid_dim=train_cfg.hid_dim,
        hid_layers=train_cfg.hid_layer,
        linear=train_cfg.linear,
        act="gelu",
    ).to(device)
    f1_teacher = copy.deepcopy(f1).to(device)
    for param in f1_teacher.parameters():
        param.requires_grad = False
    q_pred = nn.Sequential(
        nn.Linear(1, 2),
        nn.GELU(),
        nn.Linear(2, 1),
    ).to(device)
    g1 = EncoderF1(
        in_dim=1,
        out_dim=d,
        hid_dim=train_cfg.hid_dim,
        hid_layers=train_cfg.hid_layer,
        linear=train_cfg.linear,
        act="gelu",
    ).to(device)
    q_z1t1_z1t0 = GaussianCondDensityEstimator(
        in_dim=1,
        out_dim=1,
        hid_dim=train_cfg.hid_dim,
        hid_layers=train_cfg.hid_layer,
        learn_logvar=False,
    ).to(device)
    q_z1t1_xt0 = GaussianCondDensityEstimator(
        in_dim=d,
        out_dim=1,
        hid_dim=train_cfg.hid_dim,
        hid_layers=train_cfg.hid_layer,
        learn_logvar=False,
    ).to(device)
    optim_q, optim_f, optim_pred = build_optimizers(
        f1=f1,
        g1=g1,
        q_pred=q_pred,
        q_z1t1_xt0=q_z1t1_xt0,
        q_z1t1_z1t0=q_z1t1_z1t0,
        train_cfg=train_cfg,
    )
    return TrainingState(
        f1=f1,
        f1_teacher=f1_teacher,
        q_pred=q_pred,
        g1=g1,
        q_z1t1_xt0=q_z1t1_xt0,
        q_z1t1_z1t0=q_z1t1_z1t0,
        optim_q=optim_q,
        optim_f=optim_f,
        optim_pred=optim_pred,
        train_cfg=train_cfg,
        device=device,
        pred_tol=None,
        global_epoch=0,
    )


def clone_training_state(state: TrainingState) -> TrainingState:
    cloned_f1 = copy.deepcopy(state.f1).to(state.device)
    cloned_f1_teacher = copy.deepcopy(state.f1_teacher).to(state.device)
    cloned_q_pred = copy.deepcopy(state.q_pred).to(state.device)
    cloned_g1 = copy.deepcopy(state.g1).to(state.device)
    cloned_q_xt = copy.deepcopy(state.q_z1t1_xt0).to(state.device)
    cloned_q_z = copy.deepcopy(state.q_z1t1_z1t0).to(state.device)

    optim_q, optim_f, optim_pred = build_optimizers(
        f1=cloned_f1,
        g1=cloned_g1,
        q_pred=cloned_q_pred,
        q_z1t1_xt0=cloned_q_xt,
        q_z1t1_z1t0=cloned_q_z,
        train_cfg=state.train_cfg,
    )
    optim_q.load_state_dict(copy.deepcopy(state.optim_q.state_dict()))
    optim_f.load_state_dict(copy.deepcopy(state.optim_f.state_dict()))
    optim_pred.load_state_dict(copy.deepcopy(state.optim_pred.state_dict()))

    return TrainingState(
        f1=cloned_f1,
        f1_teacher=cloned_f1_teacher,
        q_pred=cloned_q_pred,
        g1=cloned_g1,
        q_z1t1_xt0=cloned_q_xt,
        q_z1t1_z1t0=cloned_q_z,
        optim_q=optim_q,
        optim_f=optim_f,
        optim_pred=optim_pred,
        train_cfg=copy.deepcopy(state.train_cfg),
        device=state.device,
        pred_tol=state.pred_tol,
        global_epoch=state.global_epoch,
    )


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_state(
    state: TrainingState,
    dataset: DatasetBundle,
    stage_id: str,
    stage_epoch: int,
    is_pre_stage: bool = False,
) -> Dict[str, float]:
    for model in (state.f1, state.f1_teacher, state.q_pred, state.g1, state.q_z1t1_xt0, state.q_z1t1_z1t0):
        model.eval()

    with torch.no_grad():
        est_z1t0 = state.f1(dataset.xt0_norm)
        est_z1t1 = state.f1(dataset.xt1_norm)
        tgt_z1t1 = state.f1_teacher(dataset.xt1_norm)
        pred_z1t1 = state.q_pred(est_z1t0)
        recon_loss = F.mse_loss(state.g1(est_z1t0), dataset.xt0_norm).item()
        jepa_loss = F.mse_loss(pred_z1t1, tgt_z1t1).item()
        logp_q_z1t1_xt0 = state.q_z1t1_xt0(x=est_z1t1, cond=dataset.xt0_norm)
        logp_q_z1t1_z1t0 = state.q_z1t1_z1t0(x=est_z1t1, cond=est_z1t0)
        cmi = (logp_q_z1t1_xt0 - logp_q_z1t1_z1t0).mean().item()
        est_std = est_z1t0.std().item()

    est_np = est_z1t0.detach().cpu().numpy().reshape(-1)
    zt0_np = dataset.raw_zt[:, 0, :]
    correlations = [safe_corrcoef(est_np, zt0_np[:, idx]) for idx in range(zt0_np.shape[1])]
    abs_corrs = np.abs(np.asarray(correlations, dtype=float))
    first_abs = abs_corrs[dataset.first_layer_indices]
    non_first_abs = abs_corrs[dataset.non_first_layer_indices] if dataset.non_first_layer_indices else np.asarray([], dtype=float)
    best_first = float(np.nanmax(first_abs)) if first_abs.size else float("nan")
    best_non_first = float(np.nanmax(non_first_abs)) if non_first_abs.size else float("nan")
    margin = best_first - best_non_first if np.isfinite(best_non_first) else best_first

    phase = "warmup" if state.global_epoch < state.train_cfg.warmup_epochs else "post_warmup"
    metrics: Dict[str, float] = {
        "epoch": state.global_epoch,
        "stage_epoch": stage_epoch,
        "stage_id": stage_id,
        "dataset_name": dataset.name,
        "phase": phase,
        "is_pre_stage": int(is_pre_stage),
        "cmi": float(cmi),
        "jepa": float(jepa_loss),
        "recon": float(recon_loss),
        "std_est_z1t0": float(est_std),
        "best_first_layer_corr": best_first,
        "best_non_first_layer_corr": best_non_first,
        "selectivity_margin": float(margin),
        "pred_tol": float(state.pred_tol) if state.pred_tol is not None else float("nan"),
        "has_nan": int(
            any(
                (not np.isfinite(v))
                for v in [cmi, jepa_loss, recon_loss, est_std, best_first, margin]
                if not isinstance(v, str)
            )
        ),
    }
    for latent_name, corr in zip(dataset.latent_names, correlations):
        metrics[f"corr_{latent_name}"] = corr
    return metrics


def subsample_for_scatter(x: np.ndarray, y: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_points, replace=False)
    return x[idx], y[idx]


def plot_scatter_snapshot(
    state: TrainingState,
    dataset: DatasetBundle,
    output_path: Path,
    title: str,
    seed: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        est_np = state.f1(dataset.xt0_norm).detach().cpu().numpy().reshape(-1)
    zt0_np = dataset.raw_zt[:, 0, :]

    n_latents = zt0_np.shape[1]
    if n_latents == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes = np.asarray(axes).reshape(-1)
    else:
        ncols = min(3, n_latents)
        nrows = int(math.ceil(n_latents / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)
        axes = np.asarray(axes).reshape(-1)

    for idx in range(len(axes)):
        ax = axes[idx]
        if idx >= n_latents:
            ax.axis("off")
            continue
        x_sub, y_sub = subsample_for_scatter(est_np, zt0_np[:, idx], state.train_cfg.scatter_points, seed + idx)
        corr = safe_corrcoef(est_np, zt0_np[:, idx])
        ax.scatter(x_sub, y_sub, s=5, alpha=0.18, linewidths=0)
        ax.set_xlabel("estimated first-layer latent")
        ax.set_ylabel(f"ground truth {dataset.latent_names[idx]}")
        ax.set_title(f"{dataset.latent_names[idx]} | corr={corr:.3f}")

    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_training_curves(metrics_df: pd.DataFrame, dataset: DatasetBundle, output_path: Path, stage_boundary: Optional[int] = None) -> None:
    corr_cols = [f"corr_{name}" for name in dataset.latent_names]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax_cmi, ax_jepa, ax_std, ax_corr = axes.reshape(-1)

    ax_cmi.plot(metrics_df["epoch"], metrics_df["cmi"], label="CMI", linewidth=1.5)
    ax_cmi.set_title("CMI vs Epoch")
    ax_cmi.set_xlabel("global epoch")
    ax_cmi.set_ylabel("cmi")

    ax_jepa.plot(metrics_df["epoch"], metrics_df["jepa"], label="JEPA", linewidth=1.5, color="tab:orange")
    ax_jepa.set_title("JEPA vs Epoch")
    ax_jepa.set_xlabel("global epoch")
    ax_jepa.set_ylabel("jepa")

    ax_std.plot(metrics_df["epoch"], metrics_df["std_est_z1t0"], label="std(est_z1t0)", linewidth=1.5, color="tab:green")
    ax_std.set_title("Estimated Latent Std")
    ax_std.set_xlabel("global epoch")
    ax_std.set_ylabel("std")

    for corr_col in corr_cols:
        ax_corr.plot(metrics_df["epoch"], metrics_df[corr_col], linewidth=1.2, label=corr_col.replace("corr_", ""))
    ax_corr.set_title("Correlation with Ground-Truth Latents")
    ax_corr.set_xlabel("global epoch")
    ax_corr.set_ylabel("corr")
    ax_corr.legend(fontsize=8, ncol=min(3, len(corr_cols)))

    if stage_boundary is not None:
        for ax in (ax_cmi, ax_jepa, ax_std, ax_corr):
            ax.axvline(stage_boundary, color="black", linestyle="--", linewidth=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_final_correlation_summary(metrics_df: pd.DataFrame, dataset: DatasetBundle, output_path: Path) -> None:
    final_row = metrics_df.iloc[-1]
    corr_vals = [final_row[f"corr_{name}"] for name in dataset.latent_names]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(dataset.latent_names, corr_vals, color=["tab:blue" if idx < dataset.n_vars_per_layer else "tab:gray" for idx in range(len(corr_vals))])
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title("Final Correlation by Latent Coordinate")
    ax.set_ylabel("corr")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_metrics_table(metrics_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)


def save_stage_summary(summary: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def save_torch_checkpoint(state: TrainingState, dataset: DatasetBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "f1": state.f1.state_dict(),
            "f1_teacher": state.f1_teacher.state_dict(),
            "q_pred": state.q_pred.state_dict(),
            "g1": state.g1.state_dict(),
            "q_z1t1_xt0": state.q_z1t1_xt0.state_dict(),
            "q_z1t1_z1t0": state.q_z1t1_z1t0.state_dict(),
            "optim_q": state.optim_q.state_dict(),
            "optim_f": state.optim_f.state_dict(),
            "optim_pred": state.optim_pred.state_dict(),
            "train_cfg": state.train_cfg.to_json_dict(),
            "pred_tol": state.pred_tol,
            "global_epoch": state.global_epoch,
            "x_mean": dataset.x_mean.detach().cpu(),
            "x_std": dataset.x_std.detach().cpu(),
        },
        output_path,
    )


def save_model_artifacts(state: TrainingState, dataset: DatasetBundle, output_dir: Path) -> None:
    save_model_bundle_npz(
        save_path=output_dir / "model_bundle.npz",
        models={
            "f1": state.f1,
            "f1_teacher": state.f1_teacher,
            "q_pred": state.q_pred,
            "g1": state.g1,
            "q_z1t1_xt0": state.q_z1t1_xt0,
            "q_z1t1_z1t0": state.q_z1t1_z1t0,
        },
        metadata={
            "x_mean": dataset.x_mean,
            "x_std": dataset.x_std,
            "pred_tol": -1.0 if state.pred_tol is None else state.pred_tol,
            "global_epoch": state.global_epoch,
            **state.train_cfg.to_json_dict(),
        },
    )
    save_torch_checkpoint(state=state, dataset=dataset, output_path=output_dir / "training_state.pt")


def compute_stability(final_metrics: Dict[str, float]) -> Tuple[bool, Dict[str, bool]]:
    checks = {
        "finite_metrics": final_metrics["has_nan"] == 0,
        "noncollapsed_std": final_metrics["std_est_z1t0"] > 1e-3,
        "positive_selectivity": final_metrics["selectivity_margin"] > 0.0,
        "first_layer_dominant": final_metrics["best_first_layer_corr"] > final_metrics["best_non_first_layer_corr"],
    }
    return all(checks.values()), checks


def train_stage(
    state: TrainingState,
    dataset: DatasetBundle,
    stage_id: str,
    output_dir: Path,
    note: Optional[str] = None,
) -> pd.DataFrame:
    ensure_dir(output_dir)
    metrics: List[Dict[str, float]] = []
    pre_metrics = evaluate_state(state=state, dataset=dataset, stage_id=stage_id, stage_epoch=0, is_pre_stage=True)
    metrics.append(pre_metrics)
    save_stage_summary(pre_metrics, output_dir / "pre_stage_metrics.json")

    snapshot_epochs = set(state.train_cfg.snapshot_epochs)
    snapshot_epochs.add(state.train_cfg.n_epochs)
    plot_scatter_snapshot(
        state=state,
        dataset=dataset,
        output_path=output_dir / "snapshots" / "epoch_0000_pre_stage.png",
        title=f"{dataset.name} | {stage_id} | pre-stage",
        seed=state.train_cfg.seed,
    )

    for local_epoch in range(1, state.train_cfg.n_epochs + 1):
        epoch_number = state.global_epoch + 1
        if epoch_number % 20 == 0:
            print(f"[{stage_id}] epoch {epoch_number}/{state.global_epoch + state.train_cfg.n_epochs}")

        state.q_z1t1_xt0.train()
        state.q_z1t1_z1t0.train()
        est_z1t0 = state.f1(dataset.xt0_norm)
        est_z1t1 = state.f1(dataset.xt1_norm)

        for _ in range(state.train_cfg.n_q_epochs):
            state.optim_q.zero_grad()
            logp_q_z1t1_xt0 = state.q_z1t1_xt0(x=est_z1t1.detach(), cond=dataset.xt0_norm)
            logp_q_z1t1_z1t0 = state.q_z1t1_z1t0(x=est_z1t1.detach(), cond=est_z1t0.detach())
            estimator_loss = -logp_q_z1t1_xt0.mean() - logp_q_z1t1_z1t0.mean()
            estimator_loss.backward()
            state.optim_q.step()

        for _ in range(state.train_cfg.n_f_epochs):
            state.optim_f.zero_grad()
            state.optim_pred.zero_grad()
            est_z1t0 = state.f1(dataset.xt0_norm)
            est_z1t1 = state.f1(dataset.xt1_norm)
            with torch.no_grad():
                tgt_z1t1 = state.f1_teacher(dataset.xt1_norm)
            pred_z1t1 = state.q_pred(est_z1t0)
            recon_loss = F.mse_loss(state.g1(est_z1t0), dataset.xt0_norm)
            mi_reg = F.mse_loss(pred_z1t1, tgt_z1t1)
            logp_q_z1t1_xt0 = state.q_z1t1_xt0(x=est_z1t1, cond=dataset.xt0_norm)
            logp_q_z1t1_z1t0 = state.q_z1t1_z1t0(x=est_z1t1, cond=est_z1t0)
            cmi_loss = (logp_q_z1t1_xt0 - logp_q_z1t1_z1t0).mean()
            var_reg = F.relu(torch.tensor(0.5, device=dataset.xt0_norm.device) - est_z1t0.std()).mean()

            if state.train_cfg.use_jepa_warmup and state.global_epoch < state.train_cfg.warmup_epochs:
                loss_f = state.train_cfg.lambda_jepa * mi_reg + state.train_cfg.lambda_recon * recon_loss
                if state.global_epoch == state.train_cfg.warmup_epochs - 1:
                    with torch.no_grad():
                        pred0 = F.mse_loss(state.q_pred(state.f1(dataset.xt0_norm)), state.f1_teacher(dataset.xt1_norm)).item()
                        state.pred_tol = 1.0 * pred0
            else:
                if state.pred_tol is None:
                    state.pred_tol = 0.0
                mi_reg = F.relu(mi_reg - state.pred_tol)
                loss_f = state.train_cfg.lambda_cmi * cmi_loss + state.train_cfg.lambda_jepa * mi_reg + 0.1 * var_reg

            loss_f.backward()
            state.optim_f.step()
            state.optim_pred.step()

            with torch.no_grad():
                for p_t, p_s in zip(state.f1_teacher.parameters(), state.f1.parameters()):
                    p_t.data.mul_(state.train_cfg.ema_m).add_(p_s.data, alpha=1.0 - state.train_cfg.ema_m)

        state.global_epoch += 1
        epoch_metrics = evaluate_state(state=state, dataset=dataset, stage_id=stage_id, stage_epoch=local_epoch, is_pre_stage=False)
        metrics.append(epoch_metrics)
        if local_epoch in snapshot_epochs:
            plot_scatter_snapshot(
                state=state,
                dataset=dataset,
                output_path=output_dir / "snapshots" / f"epoch_{local_epoch:04d}.png",
                title=f"{dataset.name} | {stage_id} | epoch {state.global_epoch}",
                seed=state.train_cfg.seed + local_epoch,
            )

    metrics_df = pd.DataFrame(metrics)
    save_metrics_table(metrics_df, output_dir / "metrics.csv")
    stage_boundary = None
    plot_training_curves(metrics_df=metrics_df, dataset=dataset, output_path=output_dir / "training_curves.png", stage_boundary=stage_boundary)
    plot_final_correlation_summary(metrics_df=metrics_df[metrics_df["is_pre_stage"] == 0], dataset=dataset, output_path=output_dir / "final_correlation_summary.png")

    final_metrics = metrics_df.iloc[-1].to_dict()
    stable, checks = compute_stability(final_metrics)
    save_stage_summary(
        {
            "stage_id": stage_id,
            "dataset_name": dataset.name,
            "note": note,
            "stable": stable,
            "checks": checks,
            "final_metrics": final_metrics,
        },
        output_dir / "stage_summary.json",
    )
    save_model_artifacts(state=state, dataset=dataset, output_dir=output_dir)
    return metrics_df


def save_dataset_inventory(specs: Sequence[DatasetSpec], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "preset": spec.preset,
            "output_path": str(spec.output_path),
            "n_layers": spec.n_layers,
            "n_vars_per_layer": spec.n_vars_per_layer,
        }
        for spec in specs
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def copy_dataset(npz_path: Path, destination_dir: Path) -> Path:
    ensure_dir(destination_dir)
    dest = destination_dir / npz_path.name
    if not dest.exists():
        dest.write_bytes(npz_path.read_bytes())
    return dest


def final_metrics_row(metrics_csv: Path) -> Dict[str, object]:
    df = pd.read_csv(metrics_csv)
    return df.iloc[-1].to_dict()


def append_experiment_cells_to_notebook(notebook_path: Path, experiment_root_expr: str) -> None:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell

    nb = nbformat.read(notebook_path, as_version=4)
    marker = "### Experiment Automation"
    if any(marker in "".join(cell.get("source", "")) for cell in nb.cells):
        return

    markdown_cell = new_markdown_cell(
        marker + "\n\nThese cells keep the original baseline notebook above intact and add helpers for staged synthetic stability runs."
    )
    code_cell = new_code_cell(
        "from pathlib import Path\n"
        "import sys\n\n"
        "NOTEBOOK_DIR = Path.cwd()\n"
        "if not (NOTEBOOK_DIR / 'first_layer_stability.py').exists():\n"
        "    NOTEBOOK_DIR = NOTEBOOK_DIR / 'synthetic_estimate_single_layer'\n"
        "sys.path.insert(0, str(NOTEBOOK_DIR))\n\n"
        "from first_layer_stability import (\n"
        "    TrainConfig,\n"
        "    append_experiment_cells_to_notebook,\n"
        "    initialize_training_state,\n"
        "    load_dataset_bundle,\n"
        "    train_stage,\n"
        ")\n\n"
        f"EXPERIMENT_ROOT = {experiment_root_expr}\n"
        "SNAPSHOT_EPOCHS = (1, 10, 25, 50, 100, 200, 500)\n"
        "SAVE_ARTIFACTS = True\n"
    )
    helper_cell = new_code_cell(
        "# Example staged helper usage\n"
        "TRAIN_CFG = TrainConfig(snapshot_epochs=SNAPSHOT_EPOCHS)\n"
        "def load_stage_dataset(npz_path, dataset_name, n_layers, n_vars_per_layer, device=device):\n"
        "    return load_dataset_bundle(Path(npz_path), dataset_name, device, n_layers, n_vars_per_layer)\n\n"
        "def init_stage_models(dataset, train_cfg=TRAIN_CFG, device=device):\n"
        "    return initialize_training_state(dataset, train_cfg, device)\n\n"
        "def run_stage(state, dataset, stage_id, run_dir):\n"
        "    return train_stage(state=state, dataset=dataset, stage_id=stage_id, output_dir=Path(run_dir))\n"
    )
    nb.cells.extend([markdown_cell, code_cell, helper_cell])
    nbformat.write(nb, notebook_path)


def write_notes_file(output_path: Path, lines: Iterable[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")
