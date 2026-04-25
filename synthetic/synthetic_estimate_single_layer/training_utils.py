from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from causallearn.utils.cit import CIT


def cross_corr_matrix(z_learned, z_true):
    if z_learned.ndim == 1:
        z_learned = z_learned[:, None]
    if z_true.ndim == 1:
        z_true = z_true[:, None]

    d_true = z_true.shape[1]

    C = np.corrcoef(z_true.T, z_learned.T)
    return C[:d_true, d_true:]


def mcc_stats_from_corr(corr: np.ndarray) -> Dict[str, float]:
    max_abs_per_true = np.max(np.abs(corr), axis=1)
    best_true_idx = int(np.argmax(max_abs_per_true))
    return {
        "mcc_mean": float(max_abs_per_true.mean()),
        "mcc_min": float(max_abs_per_true.min()),
        "mcc_max": float(max_abs_per_true.max()),
        "best_true_idx": best_true_idx,
    }


@torch.no_grad()
def kernel_ci_diag(
    model,
    dataloader: DataLoader,
    device: str,
    num_batches: int = 10,
    max_points: int = 512,   # ← controls sample size for KCI
) -> Dict[str, float]:
    """
    KCI diagnostic for:
        z_t ⟂ x_{t-1} | z_{t-1}

    Uses causallearn's KCI test.

    Returns:
        kci_p: p-value
    """
    model.eval()

    X_list = []
    Zt_list = []
    Zprev_list = []

    for b, (x_lags, x_curr, *_ ) in enumerate(dataloader):
        if b >= num_batches:
            break

        x_lags = x_lags.to(device).float()
        x_curr = x_curr.to(device).float()

        x_prev = x_lags[:, -1, :]
        x_t = x_curr.squeeze(1)

        z_prev = model.jepa.enc_online(x_prev)
        z_t = model.jepa.enc_online(x_t)

        X_list.append(x_prev.detach().cpu())
        Zt_list.append(z_t.detach().cpu())
        Zprev_list.append(z_prev.detach().cpu())

    if len(X_list) == 0:
        return {"kci_p": float("nan")}

    X = torch.cat(X_list, dim=0)
    Zt = torch.cat(Zt_list, dim=0)
    Zprev = torch.cat(Zprev_list, dim=0)

    # subsample (VERY IMPORTANT for KCI speed)
    n = X.shape[0]
    if n > max_points:
        idx = torch.randperm(n)[:max_points]
        X = X[idx]
        Zt = Zt[idx]
        Zprev = Zprev[idx]

    # convert to numpy
    X = X.numpy()
    Zt = Zt.numpy()
    Zprev = Zprev.numpy()

    # ensure 2D
    if X.ndim == 1:
        X = X[:, None]
    if Zt.ndim == 1:
        Zt = Zt[:, None]
    if Zprev.ndim == 1:
        Zprev = Zprev[:, None]

    # build data matrix: [Zt | X | Zprev]
    data = np.concatenate([Zt, X, Zprev], axis=1)

    d_z = Zt.shape[1]
    d_x = X.shape[1]
    d_cond = Zprev.shape[1]

    # indices
    i = list(range(0, d_z))                        # Zt
    j = list(range(d_z, d_z + d_x))                # X
    cond = list(range(d_z + d_x, d_z + d_x + d_cond))  # Zprev

    # run KCI
    kci_obj = CIT(data=data, method="kci")

    p_val = kci_obj(i, j, cond)

    return {
        "kci_p": float(p_val),
        "reject_0.05": bool(p_val < 0.05),
        "n_used": int(data.shape[0]),
    }