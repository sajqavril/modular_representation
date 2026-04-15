import argparse
import math
from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.data_utils import (
    seed_everything,
    split_by_traj,
    slice_by_indices,
    make_dataloader
)

from training_utils import (
    cross_corr_matrix,
    mcc_stats_from_corr,
    kernel_ci_diag,
)

# -----------------------------
# Model imports
# -----------------------------
# JEPA model
from model.model_jepa import WholeModel as JEPAWholeModel, JEPAConfig

# Flow-based CI model
from model.model import WholeModel as FlowWholeModel, LossWeights as FlowLossWeights

# Nonparametric CI / MINE / InfoNCE model
from model.model_nonparametric_ci import (
    WholeModel as MineWholeModel,
    LossWeights as MineLossWeights,
    InfoNCEModel,
)


# -----------------------------
# Shared config
# -----------------------------
@dataclass
class TrainCfg:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    seed: int = 42
    valid_ratio: float = 0.2

    # scheduling
    warmup_epochs: int = 5
    ramp_epochs: int = 10
    schedule_type: str = "cosine"

    # generic loss weights
    w_rec: float = 1.0
    w_pred: float = 1.0
    w_mi: float = 1.0
    w_cmi: float = 0.0
    w_nq0: float = 1.0
    w_nq1: float = 1.0
    w_gate_inst: float = 0.001
    w_gate_delay: float = 0.001

    # JEPA
    use_var_reg: bool = False

    # flow model
    flow_layers: int = 4
    flow_hidden: int = 64
    u_dim: int = 3

    # MINE / nonparametric model
    enc_width: int = 256
    enc_depth: int = 2
    critic_hidden: int = 256
    critic_depth: int = 2
    predictor_hidden: int = 128
    use_ema: bool = False
    alternate_training: bool = False

    # diagnostics
    run_kci: bool = False


# -----------------------------
# Scheduling
# -----------------------------
def schedule_factor(epoch: int, warmup: int, ramp: int, kind: str = "linear") -> float:
    if epoch < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    t = (epoch - warmup + 1) / ramp
    t = max(0.0, min(1.0, t))
    if kind == "linear":
        return t
    if kind == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * t))
    raise ValueError(f"Unknown schedule_type: {kind}")


# -----------------------------
# Shared evaluation
# -----------------------------
def encode_for_eval(model_type: str, model, x_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        if model_type == "jepa":
            z_raw = model.jepa.enc_online(x_tensor).detach().cpu().numpy()
        elif model_type == "flow":
            z_raw = model.encoder(x_tensor).detach().cpu().numpy()
        elif model_type in ("mine", "infonce"):
            z_raw = model.encode(x_tensor).detach().cpu().numpy()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    if z_raw.ndim == 1:
        z_raw = z_raw[:, None]
    return z_raw


def eval_mcc(
    model_type: str,
    model,
    x: np.ndarray,
    z: np.ndarray,
    cfg: TrainCfg,
    tag: str,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(x).float().to(cfg.device)
        z_true = z
        z_raw = encode_for_eval(model_type, model, x_tensor)

        corr = cross_corr_matrix(z_raw, z_true)
        stats = mcc_stats_from_corr(corr)

    corr_vec = corr[:, 0] if corr.shape[1] == 1 else None
    if corr_vec is not None:
        corr_str = " ".join([f"{c:+.3f}" for c in corr_vec.tolist()])
        print(
            f"[MCC][{tag}] mcc_mean={stats['mcc_mean']:.3f} "
            f"(min={stats['mcc_min']:.3f}, max={stats['mcc_max']:.3f}) "
            f"| best_true_idx={stats['best_true_idx']} | corr_vs_true=[{corr_str}]"
        )
    else:
        print(
            f"[MCC][{tag}] mcc_mean={stats['mcc_mean']:.3f} "
            f"(min={stats['mcc_min']:.3f}, max={stats['mcc_max']:.3f}) "
            f"| best_true_idx={stats['best_true_idx']}"
        )
        print(f"[MCC][{tag}] corr matrix:\n{corr}")

    return stats


# -----------------------------
# Model builders
# -----------------------------
def make_jepa_model(x_dim: int, z_dim: int, cfg: TrainCfg):
    jcfg = JEPAConfig(
        ema_m=0.99,
        pred_loss="mse",
        normalize=False,
        use_var_reg=cfg.use_var_reg,
        var_reg_weight=1.0,
    )
    return JEPAWholeModel(x_dim=x_dim, z_dim=z_dim, enc_width=128, enc_depth=2, jepa_cfg=jcfg)


def make_flow_model(x_dim: int, z_dim: int, window_size: int, cfg: TrainCfg):
    return FlowWholeModel(
        x_dim=x_dim,
        z_dim=z_dim,
        u_dim=cfg.u_dim,
        window_size=window_size,
        enc_width=128,
        enc_depth=2,
        dec_width=128,
        dec_depth=2,
        flow_layers=cfg.flow_layers,
        flow_hidden=cfg.flow_hidden,
        prior_type="normal",
    )


def make_mine_model(model_type: str, x_dim: int, z_dim: int, window_size: int, cfg: TrainCfg):
    if model_type == "infonce":
        return InfoNCEModel(
            x_dim=x_dim,
            z_dim=z_dim,
            window_size=window_size,
            enc_width=cfg.enc_width,
            enc_depth=cfg.enc_depth,
            predictor_hidden=cfg.predictor_hidden,
        )

    return MineWholeModel(
        x_dim=x_dim,
        z_dim=z_dim,
        window_size=window_size,
        enc_width=cfg.enc_width,
        enc_depth=cfg.enc_depth,
        critic_hidden=cfg.critic_hidden,
        critic_depth=cfg.critic_depth,
        predictor_hidden=cfg.predictor_hidden,
        use_ema=cfg.use_ema,
    )


def make_model(model_type: str, x_dim: int, z_dim: int, window_size: int, cfg: TrainCfg):
    if model_type == "jepa":
        return make_jepa_model(x_dim, z_dim, cfg)
    if model_type == "flow":
        return make_flow_model(x_dim, z_dim, window_size, cfg)
    if model_type in ("mine", "infonce"):
        return make_mine_model(model_type, x_dim, z_dim, window_size, cfg)
    raise ValueError(f"Unknown model_type: {model_type}")


# -----------------------------
# Per-model weights
# -----------------------------
def make_flow_weights(cfg: TrainCfg, epoch: int) -> FlowLossWeights:
    f = schedule_factor(epoch, cfg.warmup_epochs, cfg.ramp_epochs, cfg.schedule_type)
    return FlowLossWeights(
        w_rec=cfg.w_rec,
        w_nq0=cfg.w_nq0,
        w_nq1=cfg.w_nq1,
        w_cmi=cfg.w_cmi * f,
        w_gate_inst=cfg.w_gate_inst * f,
        w_gate_delay=cfg.w_gate_delay * f,
    )


def make_mine_weights(cfg: TrainCfg, epoch: int) -> MineLossWeights:
    f = schedule_factor(epoch, cfg.warmup_epochs, cfg.ramp_epochs, cfg.schedule_type)
    return MineLossWeights(
        w_rec=cfg.w_rec,
        w_pred=cfg.w_pred,
        w_mi=cfg.w_mi * f,
        w_cmi=cfg.w_cmi * f,
    )


# -----------------------------
# Per-model training step
# -----------------------------
def train_step_jepa(model, batch, cfg: TrainCfg, optimizer, epoch: int) -> Dict[str, float]:
    x_lags, x_curr, _, _, _ = batch
    x_lags = x_lags.to(cfg.device).float()
    x_curr = x_curr.to(cfg.device).float()

    out = model(x_lags, x_curr, w_cmi=cfg.w_cmi, mode="enc")
    loss = out["L_jepa"]

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    model.jepa.ema_update()

    return {
        "loss": float(loss.detach().cpu().item()),
        "pred": float(out["L_pred"].detach().cpu().item()),
        "var": float(out["L_var"].detach().cpu().item()),
        "cmi": float(out["cmi_batch"].detach().cpu().item()),
    }


def train_step_flow(model, batch, cfg: TrainCfg, optimizer, epoch: int) -> Dict[str, float]:
    x_lags, x_curr, _, _, _ = batch
    x_lags = x_lags.to(cfg.device).float()
    x_curr = x_curr.to(cfg.device).float()

    weights = make_flow_weights(cfg, epoch)
    out = model(x_lags, x_curr, weights=weights, mode="enc")
    loss = out["L_total"]

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)
    optimizer.step()

    stats = {
        "loss": float(loss.detach().cpu().item()),
        "L_rec": float(out["L_rec"].detach().cpu().item()) if isinstance(out["L_rec"], torch.Tensor) else float(out["L_rec"]),
        "L_nq0": float(out["L_nq0"].detach().cpu().item()) if isinstance(out["L_nq0"], torch.Tensor) else float(out["L_nq0"]),
        "L_nq1": float(out["L_nq1"].detach().cpu().item()) if isinstance(out["L_nq1"], torch.Tensor) else float(out["L_nq1"]),
        "L_cmi": float(out["L_cmi"].detach().cpu().item()) if isinstance(out["L_cmi"], torch.Tensor) else float(out["L_cmi"]),
    }
    return stats


def train_step_mine(model, batch, cfg: TrainCfg, optimizer_or_tuple, epoch: int) -> Dict[str, float]:
    x_lags, x_curr, _, _, _ = batch
    x_lags = x_lags.to(cfg.device).float()
    x_curr = x_curr.to(cfg.device).float()

    weights = make_mine_weights(cfg, epoch)

    if cfg.alternate_training and hasattr(model, "mi_critic"):
        optimizer_enc, optimizer_critic = optimizer_or_tuple

        optimizer_critic.zero_grad()
        out = model(x_lags, x_curr, weights=weights, mode="critic")
        critic_loss = out["L_mi"] + out["L_cmi"]
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.mi_critic.parameters()) + list(model.cmi_critic.parameters()),
            max_norm=1.0,
        )
        optimizer_critic.step()

        optimizer_enc.zero_grad()
        out = model(x_lags, x_curr, weights=weights, mode="enc")
        loss = out["L_total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.encoder.parameters()) + list(model.predictor.parameters()),
            max_norm=1.0,
        )
        optimizer_enc.step()
    else:
        optimizer = optimizer_or_tuple
        optimizer.zero_grad()
        out = model(x_lags, x_curr, weights=weights, mode="enc")
        loss = out["L_total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    stats = {
        "loss": float(out["L_total"].detach().cpu().item()),
        "L_pred": float(out["L_pred"].detach().cpu().item()) if "L_pred" in out else 0.0,
        "L_mi": float(out["L_mi"].detach().cpu().item()) if "L_mi" in out else 0.0,
        "L_cmi": float(out["L_cmi"].detach().cpu().item()) if "L_cmi" in out else 0.0,
        "mi_estimate": float(out["mi_estimate"].detach().cpu().item()) if "mi_estimate" in out else 0.0,
        "cmi_estimate": float(out["cmi_estimate"].detach().cpu().item()) if "cmi_estimate" in out else 0.0,
    }
    return stats


# -----------------------------
# Shared epoch loop
# -----------------------------
def make_optimizers(model_type: str, model, cfg: TrainCfg):
    if model_type == "mine" and cfg.alternate_training and hasattr(model, "mi_critic"):
        optimizer_enc = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.predictor.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        optimizer_critic = torch.optim.Adam(
            list(model.mi_critic.parameters()) + list(model.cmi_critic.parameters()),
            lr=cfg.lr,
        )
        return optimizer_enc, optimizer_critic

    return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def train_epoch(
    model_type: str,
    model,
    train_loader: DataLoader,
    cfg: TrainCfg,
    optimizers,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    logs: Dict[str, float] = {}
    num_batches = 0

    for batch in train_loader:
        if model_type == "jepa":
            out = train_step_jepa(model, batch, cfg, optimizers, epoch)
        elif model_type == "flow":
            out = train_step_flow(model, batch, cfg, optimizers, epoch)
        elif model_type in ("mine", "infonce"):
            out = train_step_mine(model, batch, cfg, optimizers, epoch)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        for k, v in out.items():
            logs[k] = logs.get(k, 0.0) + float(v)
        num_batches += 1

    for k in logs:
        logs[k] /= max(num_batches, 1)
    return logs


def print_epoch_logs(model_type: str, logs: Dict[str, float], epoch: int, epochs: int, cfg: TrainCfg):
    if model_type == "jepa":
        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"loss={logs.get('loss', 0.0):.6e} "
            f"pred={logs.get('pred', 0.0):.6e} "
            f"var={logs.get('var', 0.0):.6e} "
            f"CMI_gauss={logs.get('cmi', 0.0):.6e}"
        )
    elif model_type == "flow":
        f = schedule_factor(epoch, cfg.warmup_epochs, cfg.ramp_epochs, cfg.schedule_type)
        print(
            f"[Schedule] epoch={epoch+1}/{epochs} factor={f:.3f} | "
            f"w_rec={cfg.w_rec:.3g} w_nq0={cfg.w_nq0:.3g} w_nq1={cfg.w_nq1:.3g} "
            f"w_cmi={cfg.w_cmi * f:.3g} w_gate_inst={cfg.w_gate_inst * f:.3g} "
            f"w_gate_delay={cfg.w_gate_delay * f:.3g}"
        )
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"L_rec={logs.get('L_rec', 0.0):.4e} | "
            f"L_nq0={logs.get('L_nq0', 0.0):.4e} | "
            f"L_nq1={logs.get('L_nq1', 0.0):.4e} | "
            f"L_cmi={logs.get('L_cmi', 0.0):.4e} | "
            f"L_total={logs.get('loss', 0.0):.4e}"
        )
    elif model_type in ("mine", "infonce"):
        f = schedule_factor(epoch, cfg.warmup_epochs, cfg.ramp_epochs, cfg.schedule_type)
        print(
            f"[Schedule] epoch={epoch+1}/{epochs} factor={f:.3f} | "
            f"w_pred={cfg.w_pred:.3g} w_mi={cfg.w_mi * f:.3g} w_cmi={cfg.w_cmi * f:.3g}"
        )
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"L_pred={logs.get('L_pred', 0.0):.4e} | "
            f"L_mi={logs.get('L_mi', 0.0):.4e} | "
            f"L_cmi={logs.get('L_cmi', 0.0):.4e} | "
            f"L_total={logs.get('loss', 0.0):.4e}"
        )
        print(
            f"  MI_est={logs.get('mi_estimate', 0.0):.4f} | "
            f"CMI_est={logs.get('cmi_estimate', 0.0):.4f}"
        )


# -----------------------------
# Shared train runner
# -----------------------------
def run_training(
    model_type: str,
    model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    x_train: np.ndarray,
    z_train: np.ndarray,
    x_valid: np.ndarray,
    z_valid: np.ndarray,
    cfg: TrainCfg,
):
    model.to(cfg.device)
    optimizers = make_optimizers(model_type, model, cfg)

    for epoch in range(cfg.epochs):
        logs = train_epoch(model_type, model, train_loader, cfg, optimizers, epoch)
        print_epoch_logs(model_type, logs, epoch, cfg.epochs, cfg)

        if cfg.run_kci and model_type == "jepa":
            kci = kernel_ci_diag(
                model=model,
                dataloader=valid_loader,
                device=cfg.device,
                num_batches=10,
                max_points=512,
                num_perm=100,
            )
            print(f"  KCI_HSIC={kci['kci_hsic']:.6e} p={kci['kci_p']:.3f}")

        eval_mcc(model_type, model, x_train, z_train, cfg, tag="train")
        eval_mcc(model_type, model, x_valid, z_valid, cfg, tag="valid")

    return model


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, required=True, choices=["jepa", "flow", "mine", "infonce"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--z_dim", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.2)

    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--ramp_epochs", type=int, default=10)
    parser.add_argument("--schedule_type", type=str, default="cosine")

    parser.add_argument("--w_rec", type=float, default=1.0)
    parser.add_argument("--w_pred", type=float, default=1.0)
    parser.add_argument("--w_mi", type=float, default=1.0)
    parser.add_argument("--w_cmi", type=float, default=0.0)
    parser.add_argument("--w_nq0", type=float, default=1.0)
    parser.add_argument("--w_nq1", type=float, default=1.0)
    parser.add_argument("--w_gate_inst", type=float, default=0.001)
    parser.add_argument("--w_gate_delay", type=float, default=0.001)

    parser.add_argument("--use_var_reg", action="store_true")
    parser.add_argument("--run_kci", action="store_true")

    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--flow_hidden", type=int, default=64)
    parser.add_argument("--u_dim", type=int, default=3)

    parser.add_argument("--enc_width", type=int, default=256)
    parser.add_argument("--enc_depth", type=int, default=2)
    parser.add_argument("--critic_hidden", type=int, default=256)
    parser.add_argument("--critic_depth", type=int, default=2)
    parser.add_argument("--predictor_hidden", type=int, default=128)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--alternate_training", action="store_true")

    args = parser.parse_args()
    seed_everything(args.seed)

    cfg = TrainCfg(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
        schedule_type=args.schedule_type,
        w_rec=args.w_rec,
        w_pred=args.w_pred,
        w_mi=args.w_mi,
        w_cmi=args.w_cmi,
        w_nq0=args.w_nq0,
        w_nq1=args.w_nq1,
        w_gate_inst=args.w_gate_inst,
        w_gate_delay=args.w_gate_delay,
        use_var_reg=args.use_var_reg,
        flow_layers=args.flow_layers,
        flow_hidden=args.flow_hidden,
        u_dim=args.u_dim,
        enc_width=args.enc_width,
        enc_depth=args.enc_depth,
        critic_hidden=args.critic_hidden,
        critic_depth=args.critic_depth,
        predictor_hidden=args.predictor_hidden,
        use_ema=args.use_ema,
        alternate_training=args.alternate_training,
        run_kci=args.run_kci,
    )

    # load data
    data = np.load(args.data_path, allow_pickle=True)
    x = data["xt"]
    z = data["zt"]
    traj_id = data["traj_id"]
    B_list = data["B_list"]

    L = len(B_list)
    _, x_dim = x.shape

    print(f"Loaded data: x={x.shape}, z={z.shape}, L={L}")

    # split by trajectory
    train_idx, valid_idx = split_by_traj(traj_id, valid_ratio=cfg.valid_ratio, seed=cfg.seed)
    x_train, z_train, traj_train = slice_by_indices(x, z, traj_id, train_idx)
    x_valid, z_valid, traj_valid = slice_by_indices(x, z, traj_id, valid_idx)

    train_loader = make_dataloader(
        x=x_train,
        L=L,
        batch_size=cfg.batch_size,
        shuffle=args.shuffle,
        traj_id=traj_train,
        num_workers=0,
        drop_last=True,
        seed=cfg.seed,
    )

    valid_loader = make_dataloader(
        x=x_valid,
        L=L,
        batch_size=cfg.batch_size,
        shuffle=False,
        traj_id=traj_valid,
        num_workers=0,
        drop_last=True,
        seed=cfg.seed,
    )

    print(f"Train rows: {len(x_train)}, Valid rows: {len(x_valid)}")
    print(f"Train windows: {len(train_loader.dataset)}, Valid windows: {len(valid_loader.dataset)}")

    model = make_model(
        model_type=args.model_type,
        x_dim=x_dim,
        z_dim=args.z_dim,
        window_size=L,
        cfg=cfg,
    )

    print(f"Model type: {args.model_type}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    model = run_training(
        model_type=args.model_type,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        x_train=x_train,
        z_train=z_train,
        x_valid=x_valid,
        z_valid=z_valid,
        cfg=cfg,
    )

    print("\nFinal evaluation:")
    final_train = eval_mcc(args.model_type, model, x_train, z_train, cfg, tag="train-final")
    final_valid = eval_mcc(args.model_type, model, x_valid, z_valid, cfg, tag="valid-final")

    save_path = f"{args.model_type}_model_z{args.z_dim}_epoch{args.epochs}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
            "args": vars(args),
            "final_train_stats": final_train,
            "final_valid_stats": final_valid,
        },
        save_path,
    )
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()