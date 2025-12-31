# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def block_lower_triangular_mask(d: int, nlayer: int, device=None, dtype=None) -> torch.Tensor:
    """
    Block-lower-triangular mask (d,d) with nlayer blocks of size b=d//nlayer:
      - blocks (i,j) with i>=j are ones
      - blocks (i,j) with i<j are zeros
    """
    if d % nlayer != 0:
        raise ValueError("d must be divisible by nlayer.")
    b = d // nlayer
    Mask = torch.zeros((d, d), device=device, dtype=dtype)
    for layer in range(nlayer):
        Mask[layer*b:(layer+1)*b,:(layer+1)*b] = 1.0
    return Mask.T


class LinearModelNoMforCL(nn.Module):
    """
    z_t = sum_{tau=1..L} B_tau z_{t-tau} + eps_t
    x_t = g(z_t)
    z_t approx = g_inv(x_t)

    Progressive training:
      - active_layer decides active latent dims k = (active_layer+1)*block
      - Bs are block-lower-triangular (shared mask)
      - partial freezing via gradient masks + hard-zero inactive parts

    mask_model takes both x_lags and x_curr to compute:
      z_lags = g_inv(x_lags)
      z_curr = g_inv(x_curr)
      z_pred = sum B_tau z_{t-tau}
      eps_hat = z_curr - z_pred
      x_pred = g(z_pred)
    """
    def __init__(self, n=3, d=3, nlayer=3, L=1):
        super().__init__()
        if d % nlayer != 0:
            raise ValueError("d must be divisible by nlayer for equal-sized blocks.")
        self.d = d
        self.n = n
        self.nlayers = nlayer
        self.block = d // nlayer
        self.L = L

        self.g_inv = nn.Linear(n, d, bias=False)  # x -> z
        self.g = nn.Linear(d, n, bias=False)      # z -> x
        self.Bs = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])

        # grad masks (buffers)
        self.register_buffer("_grad_mask_ginv", torch.ones_like(self.g_inv.weight))  # (d,n)
        self.register_buffer("_grad_mask_g", torch.ones_like(self.g.weight))        # (n,d)
        self.register_buffer("_grad_mask_B", torch.ones(d, d))                      # shared across Bs

        # block-lower-triangular structure mask for Bs
        self.register_buffer("_B_block_mask", block_lower_triangular_mask(d, nlayer))

        self._hooks_installed = False
        self.init_params()

        # enforce initial structure on Bs
        with torch.no_grad():
            for B in self.Bs:
                B.weight.mul_(self._B_block_mask)

    def init_params(self):
        nn.init.xavier_uniform_(self.g_inv.weight)
        nn.init.xavier_uniform_(self.g.weight)
        for B in self.Bs:
            nn.init.normal_(B.weight, mean=0.0, std=1e-3)  # near-zero

    def _install_hooks_once(self):
        if self._hooks_installed:
            return
        self.g_inv.weight.register_hook(lambda grad: grad * self._grad_mask_ginv)
        self.g.weight.register_hook(lambda grad: grad * self._grad_mask_g)
        for B in self.Bs:
            B.weight.register_hook(lambda grad: grad * self._grad_mask_B)
        self._hooks_installed = True

    @torch.no_grad()
    def set_active_layer(self, active_layer: int):
        if not (0 <= active_layer < self.nlayers):
            raise ValueError(f"active_layer must be in [0, {self.nlayers-1}]")
        self._install_hooks_once()

        k = (active_layer + 1) * self.block

        # ---- gradient masks
        self._grad_mask_ginv.zero_()
        self._grad_mask_ginv[k-self.block:k, :] = 1.0

        self._grad_mask_g.zero_()
        self._grad_mask_g[:, k-self.block:k] = 1.0

        self._grad_mask_B.zero_()
        self._grad_mask_B[k-self.block:k, k-self.block:k] = 1.0
        self._grad_mask_B *= self._B_block_mask  # enforce structure

        # ---- hard-zero inactive params (forward safety) + enforce structure
        self.g_inv.weight[k:, :].zero_()
        self.g.weight[:, k:].zero_()
        for B in self.Bs:
            B.weight[k:, :].zero_()
            B.weight[:, k:].zero_()
            B.weight.mul_(self._B_block_mask)

    def mask_model(self, x_lags: torch.Tensor, x_curr: torch.Tensor, active_layer: int):
        """
        x_lags: (B, L, n)
        x_curr: (B, 1, n)

        returns dict with:
          z_lags: (B, L, k)
          z_pred: (B, k)
          z_curr: (B, k)
          eps_hat:(B, k)
          x_pred: (B, 1, n)
        """
        self.set_active_layer(active_layer)

        Bsz, L, n = x_lags.shape
        if L != self.L:
            raise ValueError(f"x_lags second dim should be L={self.L}, got {L}")
        if n != self.n:
            raise ValueError(f"x_lags last dim should be n={self.n}, got {n}")
        if x_curr.shape != (Bsz, 1, n):
            raise ValueError(f"x_curr must be (B,1,n)=({Bsz},1,{n}), got {tuple(x_curr.shape)}")

        k = (active_layer + 1) * self.block

        # reduced compute using sliced weights
        W_ginv = self.g_inv.weight[:k, :]   # (k, n)
        W_g = self.g.weight[:, :k]          # (n, k)

        z_lags = F.linear(x_lags, W_ginv)                  # (B, L, k)
        z_curr = F.linear(x_curr, W_ginv).squeeze(1)       # (B, k)

        z_pred = torch.zeros((Bsz, k), device=x_lags.device, dtype=x_lags.dtype)
        for tau in range(self.L):
            W_B = self.Bs[tau].weight[:k, :k]              # (k, k)
            z_pred = z_pred + F.linear(z_lags[:, tau, :], W_B)

        eps_hat = z_curr - z_pred                          # (B, k)
        x_pred = F.linear(z_curr, W_g).unsqueeze(1)         # (B, 1, n)

        return {
            "z_lags": z_lags,
            "z_pred": z_pred,
            "z_curr": z_curr,
            "eps_hat": eps_hat,
            "x_pred": x_pred,
        }

    @torch.no_grad()
    def project_Bs(self):
        """
        Call after optimizer.step() to guarantee Bs remain block-lower-triangular.
        (Especially important if you use weight decay.)
        """
        for B in self.Bs:
            B.weight.mul_(self._B_block_mask)
