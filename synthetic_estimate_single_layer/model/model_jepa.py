from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_shared import ResidualMLP

# -----------------------------
# JEPA pieces: config, predictor, module
# -----------------------------
@dataclass
class JEPAConfig:
    ema_m: float = 0.99
    pred_loss: str = "mse"      # "mse" or "cosine"
    normalize: bool = False
    use_var_reg: bool = False
    var_reg_weight: float = 1.0
    var_reg_target_std: float = 1.0
    var_reg_eps: float = 1e-4


def variance_regularization(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    if z.ndim != 2:
        z = z.reshape(z.shape[0], -1)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(target_std - std).mean()


class JEPAPredictor(nn.Module):
    def __init__(self, z_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, hidden), nn.GELU(), nn.Linear(hidden, z_dim))

    def forward(self, z):
        return self.net(z)


class SimpleJEPA(nn.Module):
    def __init__(
        self,
        encoder_online: nn.Module,
        predictor: nn.Module,
        cfg: JEPAConfig,
        encoder_target: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.enc_online = encoder_online
        self.enc_target = copy.deepcopy(encoder_online) if encoder_target is None else encoder_target
        # initialize target weights
        for pt, p in zip(self.enc_target.parameters(), self.enc_online.parameters()):
            pt.data.copy_(p.data)
        for p in self.enc_target.parameters():
            p.requires_grad_(False)
        self.pred = predictor

    @torch.no_grad()
    def ema_update(self):
        m = float(self.cfg.ema_m)
        for pt, p in zip(self.enc_target.parameters(), self.enc_online.parameters()):
            pt.data.mul_(m).add_(p.data, alpha=1.0 - m)

    def _maybe_norm(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(z, dim=-1) if self.cfg.normalize else z

    def pred_loss_fn(self, z_hat: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
        if self.cfg.pred_loss == "mse":
            return F.mse_loss(z_hat, z_tgt, reduction="mean")
        elif self.cfg.pred_loss == "cosine":
            zh = F.normalize(z_hat, dim=-1)
            zt = F.normalize(z_tgt, dim=-1)
            return (1.0 - (zh * zt).sum(dim=-1)).mean()
        else:
            raise ValueError("Unknown pred_loss")

    def forward_jepa(self, x_prev: torch.Tensor, x_curr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x_prev, x_curr: (B, x_dim)
        returns embeddings and losses
        """
        z_prev = self.enc_online(x_prev)
        z_curr = self.enc_online(x_curr)
        with torch.no_grad():
            z_tgt = self.enc_target(x_curr)

        z_prev_n = self._maybe_norm(z_prev)
        z_curr_n = self._maybe_norm(z_curr)
        z_tgt_n = self._maybe_norm(z_tgt)

        z_hat = self.pred(z_prev_n)
        L_pred = self.pred_loss_fn(z_hat, z_tgt_n)

        L_var = torch.tensor(0.0, device=z_curr.device)
        if self.cfg.use_var_reg:
            L_var = variance_regularization(z_curr_n, self.cfg.var_reg_target_std, self.cfg.var_reg_eps)
            L_var = L_var + variance_regularization(z_prev_n, self.cfg.var_reg_target_std, self.cfg.var_reg_eps)

        L_jepa = L_pred + self.cfg.var_reg_weight * L_var
        return {"z_prev": z_prev, "z_curr": z_curr, "L_pred": L_pred, "L_var": L_var, "L_jepa": L_jepa}


# -----------------------------
# WholeModel: encoder + jepa + CMI estimator
# -----------------------------
class WholeModel(nn.Module):
    def __init__(self, x_dim: int = 2, z_dim: int = 1, enc_width: int = 128, enc_depth: int = 2, jepa_cfg: Optional[JEPAConfig] = None):
        super().__init__()
        self.encoder_online = ResidualMLP(in_dim=x_dim, out_dim=z_dim, width=enc_width, depth=enc_depth)
        self.encoder_target = copy.deepcopy(self.encoder_online)
        self.predictor = JEPAPredictor(z_dim, hidden=max(32, enc_width // 4))
        self.jepa_cfg = jepa_cfg or JEPAConfig()
        self.jepa = SimpleJEPA(self.encoder_online, self.predictor, self.jepa_cfg, encoder_target=self.encoder_target)
        self.z_dim = z_dim

    def forward(self, x_lags: torch.Tensor, x_curr: torch.Tensor, w_cmi: float = 0.0, mode: str = "enc"):
        """
        x_lags: (B, L, x_dim), x_curr: (B, 1, x_dim)
        Returns dict with JEPA losses and CMI estimate on the batch
        """
        batch_size = x_lags.shape[0]
        x_prev = x_lags[:, -1, :]            # x_{t-1}
        x_t = x_curr.squeeze(1)              # x_t

        jepa_out = self.jepa.forward_jepa(x_prev=x_prev, x_curr=x_t)
        z_t = jepa_out["z_curr"]             # (B, z_dim)
        z_t_minus1 = self.jepa.enc_online(x_prev).detach()  # (B, z_dim) - treat as previous embedding (no grad)

        # Conditional MI estimate (linear Gaussian approx) on this minibatch
        cmi_val = self.compute_conditional_mi_batch(z_t.detach(), z_t_minus1.detach(), x_prev.detach())

        # Combine losses
        L_jepa = jepa_out["L_jepa"]
        L_total = L_jepa + w_cmi * cmi_val  # minimize JEPA + small CMI penalty if desired

        out = {
            "L_jepa": L_jepa,
            "L_pred": jepa_out["L_pred"],
            "L_var": jepa_out["L_var"],
            "L_total": L_total,
            "cmi_batch": cmi_val,
            "z_t": z_t.detach(),
            "z_t_minus1": z_t_minus1.detach()
        }
        return out

    @staticmethod
    def compute_conditional_mi_batch(z_t: torch.Tensor, z_t_minus1: torch.Tensor, x_t_minus1: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
        """
        Linear-Gaussian conditional MI estimator:
          I(z_t ; x_{t-1} | z_{t-1}) = 0.5 * (log det Cov(resid from z_{t-1}) - log det Cov(resid from z_{t-1}, x_{t-1}))
        We estimate residual covariances from linear least squares on the minibatch.
        z_t: (B, d_z), z_t_minus1: (B, d_z), x_t_minus1: (B, x_dim)
        Returns scalar tensor (nats)
        """
        B = z_t.shape[0]
        device = z_t.device
        # Form design matrices
        # regress z_t on z_{t-1} -> residuals r1
        A = z_t_minus1  # (B, d_z)
        # solve for W in least squares: A W^T ≈ z_t
        # add constant bias
        A_aug = torch.cat([A, torch.ones(B, 1, device=device)], dim=1)  # (B, d_z+1)
        sol1 = torch.linalg.lstsq(A_aug, z_t).solution  # (d_z+1, d_z)
        pred1 = A_aug @ sol1  # (B, d_z)
        r1 = z_t - pred1
        # covariance of residuals
        C1 = (r1 - r1.mean(0)).t() @ (r1 - r1.mean(0)) / max(B - 1, 1)
        # regress z_t on [z_{t-1}, x_{t-1}]
        Bmat = torch.cat([A, x_t_minus1], dim=1)
        B_aug = torch.cat([Bmat, torch.ones(B, 1, device=device)], dim=1)
        sol2 = torch.linalg.lstsq(B_aug, z_t).solution
        pred2 = B_aug @ sol2
        r2 = z_t - pred2
        C2 = (r2 - r2.mean(0)).t() @ (r2 - r2.mean(0)) / max(B - 1, 1)

        # Regularize covariances for numerical stability
        eps = ridge
        C1 = C1 + eps * torch.eye(C1.shape[0], device=device)
        C2 = C2 + eps * torch.eye(C2.shape[0], device=device)
        # logdet via slogdet
        sign1, ld1 = torch.slogdet(C1)
        sign2, ld2 = torch.slogdet(C2)
        # If determinants non-positive (should not with ridge), clamp
        if (sign1 <= 0).any() or (sign2 <= 0).any():
            # fallback to trace-based surrogate small positive
            return torch.tensor(0.0, device=device)
        cmi = 0.5 * (ld1 - ld2)
        # ensure scalar
        if cmi.numel() > 1:
            cmi = cmi.sum()
        # clamp to >= 0
        return torch.clamp(cmi, min=0.0)