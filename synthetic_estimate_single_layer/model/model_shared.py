# model.py
from __future__ import annotations

import torch
import torch.nn as nn


# -----------------------------
# Simple Residual MLP encoder
# -----------------------------
class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 256, depth: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.GELU()) for _ in range(depth)])
        self.out = nn.Linear(width, out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)