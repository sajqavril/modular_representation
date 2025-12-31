# utils.py
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# utils.py (or seed.py)
import os
import random
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True):
    """
    Seed all random number generators for full reproducibility.

    Parameters
    ----------
    seed : int
        Global seed value
    deterministic : bool
        If True, forces deterministic CUDA behavior (slower)
    """
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # # Hash seed (important for some Python internals)
    # os.environ["PYTHONHASHSEED"] = str(seed)

    # if deterministic:
    #     # cuDNN / CUDA determinism
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    #     # PyTorch >= 1.8
    #     try:
    #         torch.use_deterministic_algorithms(True)
    #     except Exception:
    #         pass

def seed_worker(worker_id):
    """
    Ensures each DataLoader worker has a deterministic seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def generate_multiview_x_datasets(
    z: np.ndarray,
    A: np.ndarray,
    save_dir: str,
    save_npz: bool = False,
) -> Dict[int, np.ndarray]:
    """
    For k = 1..d:
      x^(k)_t = A[:, :k] @ z_t[:k]
    Returns dict: k -> x^(k) with shape (T, d)
    """
    os.makedirs(save_dir, exist_ok=True)

    T, d = z.shape
    assert A.shape == (d, d)

    x_dict: Dict[int, np.ndarray] = {}
    for k in range(1, d + 1):
        z_k = z[:, :k]
        A_k = A[:, :k]
        x_k = z_k @ A_k.T
        x_dict[k] = x_k.astype(np.float32)

        if save_npz:
            path = os.path.join(save_dir, f"x_k{k}.npz")
            np.savez(path, x=x_k.astype(np.float32), A=A_k.astype(np.float32), z=z_k.astype(np.float32), k=k)

    return x_dict


class LagWindowDataset(Dataset):
    """
    Given x (T,n), yields:
      x_lags: (L,n) = x[t-L:t]
      x_curr: (1,n) = x[t:t+1]
      t_idx:  int   = t
    for t = L .. T-1
    """
    def __init__(self, x: np.ndarray, L: int):
        if x.ndim != 2:
            raise ValueError("x must have shape (T,n)")
        if L < 1:
            raise ValueError("L must be >= 1")
        self.x = x.astype(np.float32)
        self.L = L
        self.T, self.n = self.x.shape
        if self.T <= L:
            raise ValueError(f"T must be > L. Got T={self.T}, L={L}")

    def __len__(self):
        return self.T - self.L

    def __getitem__(self, idx: int):
        t = idx + self.L
        x_lags = self.x[t - self.L:t, :]
        x_curr = self.x[t:t + 1, :]
        return (
            torch.from_numpy(x_lags),          # (L,n)
            torch.from_numpy(x_curr),          # (1,n)
            torch.tensor(t, dtype=torch.long)  # scalar
        )


def make_dataloader_for_k(
    x_dict: Dict[int, np.ndarray],
    k: int,
    L: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    drop_last: bool = True,
    seed: int = 42
) -> DataLoader:
    x = x_dict[k]
    ds = LagWindowDataset(x, L=L)
    g = torch.Generator().manual_seed(seed)
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        generator=g,
        worker_init_fn=seed_worker
    )


def offdiag_mean_abs(M: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute value of off-diagonal entries.
    M: (k,k)
    """
    k = M.shape[0]
    off = M - torch.diag(torch.diag(M))
    return off.abs().sum() / (k * (k - 1) + 1e-12)


def whiteness_loss(
    eps_hat: torch.Tensor,
    t_idx: Optional[torch.Tensor] = None,
    lambda_offdiag: float = 1.0,
    lambda_lag1: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    """
    Whiteness / independence-ish penalties for eps_hat (B,k).

    1) Cross-dimension decorrelation: penalize off-diagonal of covariance.
    2) Temporal lag-1 autocorrelation: if t_idx is provided, estimate corr between eps(t) and eps(t-1)
       using adjacent time indices found inside the minibatch.

    Notes:
    - If you use shuffle=True, temporal adjacency in-batch is rare; lag1 term will be ~0 (or skipped).
      For lag1 penalty, use shuffle=False or a custom sampler.
    """
    B, k = eps_hat.shape
    e0 = eps_hat - eps_hat.mean(dim=0, keepdim=True)

    # ---- (1) off-diagonal covariance penalty
    cov = (e0.T @ e0) / (B - 1 + eps)  # (k,k)
    # normalize to correlation-like to reduce scale sensitivity
    var = torch.diag(cov).clamp_min(eps)
    std = torch.sqrt(var)
    corr = cov / (std[:, None] * std[None, :] + eps)
    loss_offdiag = offdiag_mean_abs(corr)

    # ---- (2) lag-1 autocorrelation penalty (optional)
    loss_lag1 = eps_hat.new_tensor(0.0)
    used_pairs = 0

    if t_idx is not None:
        # sort by time within batch
        order = torch.argsort(t_idx)
        t_sorted = t_idx[order]
        e_sorted = e0[order]  # centered

        # find adjacent time pairs where t[i] = t[i-1] + 1
        dt = t_sorted[1:] - t_sorted[:-1]
        mask = (dt == 1)
        if mask.any():
            e_t = e_sorted[1:][mask]      # (P,k)
            e_tm1 = e_sorted[:-1][mask]   # (P,k)
            used_pairs = int(e_t.shape[0])

            # per-dim correlation
            num = (e_t * e_tm1).mean(dim=0)
            den = torch.sqrt((e_t.pow(2).mean(dim=0) * e_tm1.pow(2).mean(dim=0)).clamp_min(eps))
            corr1 = num / (den + eps)     # (k,)
            loss_lag1 = corr1.abs().mean()

    total = lambda_offdiag * loss_offdiag + lambda_lag1 * loss_lag1
    stats = {
        "loss_offdiag": float(loss_offdiag.detach().cpu().item()),
        "loss_lag1": float(loss_lag1.detach().cpu().item()),
        "lag1_pairs": used_pairs,
    }
    return total, stats
