import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class LagWindowDataset(Dataset):
    """
    Multi-trajectory aware lag-window dataset.

    Inputs:
      x:        (T_total, n)
      traj_id:  (T_total,) integer trajectory id for each row in x
      L:        lag length

    Returns:
      x_lags:   (L, n)
      x_curr:   (1, n)
      t_global: scalar
      traj:     scalar
      t_local:  scalar
    """
    def __init__(self, x: np.ndarray, L: int, traj_id: Optional[np.ndarray] = None):
        if x.ndim != 2:
            raise ValueError("x must have shape (T_total, n)")
        if L < 1:
            raise ValueError("L must be >= 1")

        self.x = np.asarray(x, dtype=np.float32)
        self.L = int(L)
        self.T_total, self.n = self.x.shape

        if traj_id is None:
            traj_id = np.zeros((self.T_total,), dtype=np.int64)
        traj_id = np.asarray(traj_id, dtype=np.int64)

        if traj_id.shape != (self.T_total,):
            raise ValueError(f"traj_id must have shape (T_total,), got {traj_id.shape}")

        self.traj_id = traj_id

        valid_t = []
        local_t = []

        start = 0
        while start < self.T_total:
            tid = self.traj_id[start]
            end = start
            while end < self.T_total and self.traj_id[end] == tid:
                end += 1

            seg_len = end - start
            if seg_len > self.L:
                for t in range(start + self.L, end):
                    valid_t.append(t)
                    local_t.append(t - start)

            start = end

        if len(valid_t) == 0:
            raise ValueError("No valid windows found. Check L and trajectory lengths.")

        self.valid_t = np.asarray(valid_t, dtype=np.int64)
        self.local_t = np.asarray(local_t, dtype=np.int64)

    def __len__(self):
        return int(self.valid_t.shape[0])

    def __getitem__(self, idx: int):
        t = int(self.valid_t[idx])
        tl = int(self.local_t[idx])
        tid = int(self.traj_id[t])

        x_lags = self.x[t - self.L:t, :]
        x_curr = self.x[t:t + 1, :]

        return (
            torch.from_numpy(x_lags),
            torch.from_numpy(x_curr),
            torch.tensor(t, dtype=torch.long),
            torch.tensor(tid, dtype=torch.long),
            torch.tensor(tl, dtype=torch.long),
        )


def split_by_traj(
    traj_id: np.ndarray,
    valid_ratio: float = 0.2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split row indices by whole trajectories.

    Returns:
      train_idx: row indices belonging to train trajectories
      valid_idx: row indices belonging to valid trajectories
    """
    traj_id = np.asarray(traj_id, dtype=np.int64)
    uniq = np.unique(traj_id)

    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n_valid = max(1, int(len(uniq) * valid_ratio))
    valid_trajs = set(uniq[:n_valid])
    train_trajs = set(uniq[n_valid:])

    train_idx = np.where(np.isin(traj_id, list(train_trajs)))[0]
    valid_idx = np.where(np.isin(traj_id, list(valid_trajs)))[0]
    return train_idx, valid_idx


def slice_by_indices(
    x: np.ndarray,
    z: np.ndarray,
    traj_id: np.ndarray,
    idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice arrays by raw row indices and re-map trajectory ids to contiguous 0..K-1.
    """
    x_sub = x[idx]
    z_sub = z[idx]
    traj_sub = traj_id[idx]

    uniq = np.unique(traj_sub)
    remap = {tid: i for i, tid in enumerate(uniq)}
    traj_sub = np.array([remap[t] for t in traj_sub], dtype=np.int64)

    return x_sub, z_sub, traj_sub


def make_dataloader(
    x: np.ndarray,
    L: int,
    batch_size: int,
    shuffle: bool,
    traj_id: Optional[np.ndarray] = None,
    num_workers: int = 0,
    drop_last: bool = True,
    seed: int = 42,
) -> DataLoader:
    ds = LagWindowDataset(x=x, L=L, traj_id=traj_id)
    g = torch.Generator().manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g,
    )