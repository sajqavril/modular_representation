# -*- coding: utf-8 -*-
"""
Gated conditional flow model for:
- z_t = f(x_t), z_{t+1} = f(x_{t+1})
- q0(z_{t+1} | z_t): conditional autoregressive flow
- q1(z_{t+1} | z_t, u_t): conditional autoregressive flow (extra info from x_t via u_t)
- Noise prior p_E(e) with configurable distribution family
Losses:
- L_rec: reconstruction loss on x_t
- L_noise: -log q0(z_{t+1}|z_t)
- L_cmi:  log q1(z_{t+1}|z_t,u_t) - log q0(z_{t+1}|z_t)
Optional:
- L_sparse_inst / L_sparse_delay on learned gates
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Literal

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from synthetic_estimate_single_layer.data.data_utils import stable_softplus, batch_standardize, lowpass_fft_coarsen

def gaussian_nll_diag(y: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """
        Diagonal Gaussian negative log-likelihood, averaged over batch.
        y, mu, log_sigma: (B, D)
        Returns scalar.
        """
        # clamp log_sigma for stability
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=10.0)
        inv_var = torch.exp(-2.0 * log_sigma)  # 1/sigma^2
        # 0.5 * [ (y-mu)^2 / sigma^2 + 2 log sigma + log(2pi) ]
        nll = 0.5 * ((y - mu) ** 2 * inv_var + 2.0 * log_sigma + math.log(2.0 * math.pi))
        return nll.sum(dim=-1).mean()

# -----------------------------
# Noise prior p_E(e)
# -----------------------------

class NoisePrior(nn.Module):
    """Base class: factorized noise prior p_E(e)=∏_i p(e_i)."""
    def log_prob(self, e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e: (batch_size, d)
        Returns:
            log p_E(e): (batch_size,)
        """
        raise NotImplementedError

class StandardNormalPrior(NoisePrior):
    def log_prob(self, e: torch.Tensor) -> torch.Tensor:
        # sum over dims
        if e.ndim == 1:
            return -0.5 * (e**2) - 0.5 * math.log(2 * math.pi)
        elif e.ndim == 2:
            return -0.5 * (e**2).sum(dim=-1)
            retun - F.mse_loss() #- 0.5 * math.log(2 * math.pi) * e.shape[1]
        else:
            raise NotImplementedError
        # return (-0.5 * (e ** 2) - 0.5 * math.log(2 * math.pi)).sum(dim=-1)


class StandardLogisticPrior(NoisePrior):
    def log_prob(self, e: torch.Tensor) -> torch.Tensor:
        # Logistic(0,1): log pdf = -e - 2*softplus(-e)
        # stable form:
        return (-e - 2.0 * F.softplus(-e)).sum(dim=-1)


def make_prior(prior_type: Literal["normal", "logistic"],
               layer_dim: int,
               **kwargs) -> NoisePrior:
    if prior_type == "normal":
        return StandardNormalPrior()
    if prior_type == "logistic":
        return StandardLogisticPrior()
    raise ValueError(f"Unknown prior_type={prior_type}")


# -----------------------------
# Encoder / Decoder 
# -----------------------------

class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 512, depth: int = 6,
                 dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, width)
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        self.dropout = dropout

        for _ in range(depth):
            self.norms1.append(nn.BatchNorm1d(width))
            self.norms2.append(nn.BatchNorm1d(width))
            self.fcs1.append(nn.Linear(width, width))
            self.fcs2.append(nn.Linear(width, width))

        self.out_norm = nn.BatchNorm1d(width)
        self.out_proj = nn.Linear(width, out_dim)

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier/Glorot for GELU
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for n1, n2, fc1, fc2 in zip(self.norms1, self.norms2, self.fcs1, self.fcs2):
            r = h #n1(h)
            r = F.relu(fc1(r))
            # r = n2(r)
            r = fc2(r)
            if self.dropout > 0:
                r = F.dropout(r, p=self.dropout, training=self.training)
            h = h + r
        # h = self.out_norm(h)
        return self.out_proj(h)

# # -----------------------------
# # InfoNCE for MI maximization (not recon)
# # -----------------------------

# class SimpleInfoNCEEncoder(nn.Module):
#     def __init__(self, x_dim: int, z_dim: int, temperature: float = 1.0):
#         super().__init__()
#         self.temperature = temperature
#         # Just ONE linear layer, no normalization, no residuals
#         self.encoder = nn.Sequential(
#             nn.Linear(x_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, z_dim)
#         )
        
#         # Proper initialization
#         for m in self.encoder.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         z = self.encoder(x)
#         z = F.normalize(z, dim=-1, p=2)
#         return z
    
#     def infonce_loss(self, x_t0: torch.Tensor, x_t1: torch.Tensor):
#         z0 = self.forward(x_t0)
#         z1 = self.forward(x_t1)
        
#         # Simple InfoNCE
#         logits = (z0 @ z1.t()) / self.temperature  # (B, B)
#         labels = torch.arange(z0.size(0), device=z0.device)
#         loss = F.cross_entropy(logits, labels)
        
#         with torch.no_grad():
#             acc = (logits.argmax(dim=1) == labels).float().mean()
        
#         return loss, {"acc_inbatch": acc, "tau": torch.tensor(self.temperature)}
    
class AugmentedInfoNCEEncoder(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, temperature: float = 0.1, 
                 aug_noise: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.aug_noise = aug_noise
        
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, z_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        # F.normalize(z, dim=-1, p=2) # DON'T USE THIS WHEN Z_DIM=1!!!!!!!
        return z
    
    def augment(self, x):
        """Add noise augmentation - creates different 'views' of same x"""
        return x + torch.randn_like(x) * self.aug_noise
    
    def infonce_loss(self, x_batch):
        """
        x_batch: (B, x_dim) - just need ONE set of samples
        Create two augmented views of the SAME samples
        """
        # Two different augmentations of the same data
        x_aug1 = self.augment(x_batch)
        x_aug2 = self.augment(x_batch)
        
        z1 = self(x_aug1)
        z2 = self(x_aug2)
        
        # Symmetric InfoNCE
        logits_12 = (z1 @ z2.t()) / self.temperature
        logits_21 = (z2 @ z1.t()) / self.temperature
        
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = 0.5 * (F.cross_entropy(logits_12, labels) + 
                      F.cross_entropy(logits_21, labels))
        
        with torch.no_grad():
            acc = (logits_12.argmax(dim=1) == labels).float().mean()
        
        return loss, {"acc_inbatch": acc}

# class InfoNCEEncoder(nn.Module):
#     """
#     Drop-in encoder with InfoNCE:
#       - forward(x) -> z  (used by your flows)
#       - infonce_loss(x0, x1) -> scalar loss + stats

#     Uses a momentum target encoder phi for stability.
#     """
#     def __init__(
#         self,
#         x_dim: int,
#         z_dim: int,
#         width: int = 512,
#         depth: int = 6,
#         dropout: float = 0.0,
#         temperature: float = 0.1,
#         normalize: bool = True,
#         use_momentum_target: bool = True,
#         momentum: float = 0.99,
#         symmetric: bool = True,
#     ):
#         super().__init__()
#         self.temperature = float(temperature)
#         self.normalize = bool(normalize)
#         self.use_momentum_target = bool(use_momentum_target)
#         self.momentum = float(momentum)
#         self.symmetric = bool(symmetric)

#         # online encoder h1
#         self.online = ResidualMLP(x_dim, z_dim, width=width, depth=depth, dropout=dropout)

#         # target encoder phi (EMA copy of online)
#         self.target = ResidualMLP(x_dim, z_dim, width=width, depth=depth, dropout=dropout)
        
#         with torch.no_grad():
#             for m in self.online.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.xavier_uniform_(m.weight, gain=0.01)  # Smaller init
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
        
#         self._init_target()

#         if self.use_momentum_target:
#             for p in self.target.parameters():
#                 p.requires_grad_(False)

#     @torch.no_grad()
#     def _init_target(self):
#         for p_t, p in zip(self.target.parameters(), self.online.parameters()):
#             p_t.data.copy_(p.data)

#     @torch.no_grad()
#     def update_target(self):
#         """Call once per optimizer step."""
#         if not self.use_momentum_target:
#             return
#         m = self.momentum
#         for p_t, p in zip(self.target.parameters(), self.online.parameters()):
#             p_t.data.mul_(m).add_(p.data, alpha=1.0 - m)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Drop-in replacement for old encoder: returns z."""
#         z = self.online(x)
#         return z

#     def _embed_online(self, x: torch.Tensor) -> torch.Tensor:
#         z = self.online(x)
#         if self.normalize:
#             z = F.normalize(z, dim=-1)
#         return z

#     # @torch.no_grad()
#     def _embed_target(self, x: torch.Tensor) -> torch.Tensor:
#         z = self.target(x)
#         if self.normalize:
#             z = F.normalize(z, dim=-1)
#         return z

#     def _rec_from_embeddings(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
#         # logits_ij = sim(z0_i, z1_j) / tau
#         logits = (z0 @ z1.t()) / self.temperature  # (B, B)
#         labels = torch.arange(z0.size(0), device=z0.device)
#         return F.cross_entropy(logits, labels)

#     def infonce_loss(self, x_t0: torch.Tensor, x_t1: torch.Tensor):
#         z0 = self._embed_online(x_t0)
#         z1 = self._embed_online(x_t1)  # Both use online now
        
#         loss01 = self._rec_from_embeddings(z0, z1)
        
#         if self.symmetric:
#             loss10 = self._rec_from_embeddings(z1, z0)
#             loss = 0.5 * (loss01 + loss10)
#         else:
#             loss = loss01
        
#         with torch.no_grad():
#             logits = (z0 @ z1.t()) / self.temperature
#             acc = (logits.argmax(dim=1) == torch.arange(z0.size(0), device=z0.device)).float().mean()
        
#         return loss, {"acc_inbatch": acc, "tau": torch.tensor(self.temperature, device=x_t0.device)}

# -----------------------------
# Gated conditional autoregressive flow (MAF-style layer)
# -----------------------------

class GatedConditionalMAFLayer(nn.Module):
    """
    One conditional MAF layer:
      e_i = (z_i - mu_i(z_{<i}, z_lagged, context)) / sigma_i(z_{<i}, z_lagged, context)

    Gates determine which inputs contribute to each output dimension.
    Each dimension has its own network, but all computed in parallel.
    """

    def __init__(self, layer_dim: int, window_size: int, context_dim: int, hidden: int = 512):
        super().__init__()
        self.layer_dim = layer_dim
        self.window_size = window_size
        self.context_dim = context_dim
        self.hidden = hidden
        self.hidden_per_dim = hidden // layer_dim  # hidden units per dimension

        # Learnable gates:
        # alpha_inst[i,j]: does z_j (where j < i) contribute to mu_i?
        self.alpha_inst = nn.Parameter(torch.zeros(self.layer_dim, self.layer_dim))
        
        # beta_delay[i,k]: does z_lagged[k] contribute to mu_i?
        self.beta_delay = nn.Parameter(torch.zeros(self.layer_dim, self.layer_dim * self.window_size))

        # Separate network for EACH dimension
        # Input per dimension: z_t_gated (layer_dim) + z_lagged_gated (d*w) + context (layer_dim)
        input_per_dim = self.layer_dim + (self.layer_dim * self.window_size) + self.context_dim
        
        # Create separate networks using ModuleList
        self.networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_per_dim, self.hidden_per_dim),
                nn.GELU(),
                nn.Linear(self.hidden_per_dim, self.hidden_per_dim),
                nn.GELU(),
                nn.Linear(self.hidden_per_dim, 2)  # outputs [mu_i, log_sigma_i]
            )
            for _ in range(self.layer_dim)
        ])
        with torch.no_grad():
            for net in self.networks:
                net[-1].weight.data *= 0.1  # Small weights
                net[-1].bias.data[0] = 0.0   # mu starts at 0
                net[-1].bias.data[1] = 0.0   # log_sigma starts at 0 (sigma=1)
        

    def _instantaneous_gates_matrix(self) -> torch.Tensor:
        """Strictly lower triangular gates with temperature control"""
        g = stable_softplus(self.alpha_inst)
        # Clip to prevent extreme values
        g = torch.clamp(g, max=10.0)
        return torch.tril(g, diagonal=-1)

    def _delayed_gates_matrix(self) -> torch.Tensor:
        """Full positive gates with temperature control"""
        g = stable_softplus(self.beta_delay)
        # Clip to prevent extreme values
        g = torch.clamp(g, max=10.0)
        return g


    def forward_z_to_e(self,
                    z_t: torch.Tensor,
                    z_lagged: torch.Tensor,
                    context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute all dimensions in parallel using separate networks.
        
        Args:
            z_t: (batch_size, layer_dim)          
            z_lagged: (batch_size, layer_dim, window_size)
            context: (batch_size, context_dim)
        Returns:
            e_t: (batch_size, layer_dim)
            log_det: (batch_size,)
        """
        batch_size = z_t.shape[0]
        
        # Get gate matrices
        M = self._instantaneous_gates_matrix()     # (d, d) lower triangular
        Bm = self._delayed_gates_matrix()          # (d, d*w)
        
        # Flatten z_lagged: (B, d*w)
        z_lag_flat = z_lagged.reshape(batch_size, -1)
        
        # --- Vectorized gating for all output dimensions ---
        
        # Expand z_t for all output dimensions: (B, d, d)
        z_t_expanded = z_t.unsqueeze(1).expand(-1, self.layer_dim, -1)
        
        # Apply instantaneous gates: (B, d, d)
        # z_inst_gated[b, i, j] = z_t[b, j] * M[i, j]
        z_inst_gated = z_t_expanded * M.unsqueeze(0)
        
        # Expand z_lagged for all output dimensions: (B, d, d*w)
        z_lag_expanded = z_lag_flat.unsqueeze(1).expand(-1, self.layer_dim, -1)
        
        # Apply delayed gates: (B, d, d*w)
        z_delay_gated = z_lag_expanded * Bm.unsqueeze(0)
        
        # Expand context for all output dimensions: (B, d, d)
        context_expanded = context.unsqueeze(1).expand(-1, self.layer_dim, -1)
        
        # Concatenate all inputs for each dimension: (B, d, input_per_dim)
        x_in = torch.cat([z_inst_gated, z_delay_gated, context_expanded], dim=-1)
        
        # Process each dimension through its own network
        # Simple loop over dimensions (not batches, so it's fast)
        outputs = []
        for i in range(self.layer_dim):
            # x_in[:, i, :] has shape (B, input_per_dim)
            out_i = self.networks[i](x_in[:, i, :])  # (B, 2)
            outputs.append(out_i)
        
        # Stack all outputs: list of (B, 2) -> (B, d, 2)
        out = torch.stack(outputs, dim=1)
        
        # Extract mu and log_sigma
        raw_mu = out[:, :, 0]           # (B, d)
        raw_log_sigma = out[:, :, 1]
        mu = raw_mu                      # no tanh
        # log_sigma = raw_log_sigma
        max_abs = 5.0
        log_sigma = max_abs * torch.tanh(raw_log_sigma / max_abs) #raw_log_sigma.clamp(-5.0, 5.0)
        sigma = torch.exp(log_sigma)
        # if torch.rand(()) < 0.001:  # sample occasionally
        #     print("raw_log_sigma", raw_log_sigma.min().item(), raw_log_sigma.max().item())
        #     print("log_sigma", log_sigma.min().item(), log_sigma.max().item())
        #     print("mu", raw_mu.min().item(), raw_mu.max().item())
        log_det = (-log_sigma).sum(-1)
        e = (z_t - mu) #/ sigma
        # mu = 10.0 * torch.tanh(raw_mu)
        # log_sigma = 2.0 * F.tanh(raw_log_sigma)       # hard floor
        # sigma = torch.exp(log_sigma)
        # log_det = (-log_sigma).sum(-1)
        # e = (z_t - mu) / sigma

        # clamp log det to prevent it from dominating
        # log_det_capped = torch.clamp(log_det, min=-10, max=10)
    
        return e, 0. #log_det

    def log_prob(self,
                 z_t: torch.Tensor,
                 z_lagged: torch.Tensor,
                 context: torch.Tensor,
                 prior: NoisePrior) -> torch.Tensor:
        """
        Compute log q(z | z_lagged, context) for this layer.
        """
        e, log_det = self.forward_z_to_e(z_t=z_t, z_lagged=z_lagged, context=context)
        log_prior = prior.log_prob(e)
        
        # Check for NaN/Inf
        if torch.isnan(log_det).any() or torch.isinf(log_det).any():
            print(f"Warning: Invalid log_det values detected")
        if torch.isnan(log_prior).any() or torch.isinf(log_prior).any():
            print(f"Warning: Invalid log_prior values detected")
            
        return log_prior + 0 * log_det # to debug the logdet term

    def gate_penalties(self) -> Dict[str, torch.Tensor]:
        """Compute L1 penalties on gate values for sparsity"""
        A = self._instantaneous_gates_matrix()
        Bm = self._delayed_gates_matrix()
        l1_inst = A.abs().sum()
        l1_delay = Bm.abs().sum()
        return {"l1_inst": l1_inst, "l1_delay": l1_delay}

class SimpleGatedConditionalMAFLayer(nn.Module):
    """
    One conditional MAF layer:
      e_i = (z_i - mu_i(z_{<i}, z_lagged, context)) / sigma_i(z_{<i}, z_lagged, context)

    Gates determine which inputs contribute to each output dimension.
    Each dimension has its own network, but all computed in parallel.
    """

    def __init__(self, layer_dim: int, window_size: int, context_dim: int, hidden: int = 512):
        super().__init__()
        self.layer_dim = layer_dim
        self.window_size = window_size
        self.context_dim = context_dim
        self.hidden = hidden
        self.hidden_per_dim = hidden // layer_dim  # hidden units per dimension

        # Learnable gates:
        # alpha_inst[i,j]: does z_j (where j < i) contribute to mu_i?
        self.alpha_inst = nn.Parameter(torch.zeros(self.layer_dim, self.layer_dim))
        
        # beta_delay[i,k]: does z_lagged[k] contribute to mu_i?
        self.beta_delay = nn.Parameter(torch.zeros(self.layer_dim, self.layer_dim * self.window_size))

        # Separate network for EACH dimension
        # Input per dimension: z_t_gated (layer_dim) + z_lagged_gated (d*w) + context (layer_dim)
        input_per_dim = self.layer_dim + (self.layer_dim * self.window_size) + self.context_dim
        
        # Create separate networks using ModuleList
        self.net = nn.Sequential(
                nn.Linear(input_per_dim, self.hidden_per_dim),
                nn.GELU(),
                nn.Linear(self.hidden_per_dim, self.hidden_per_dim),
                nn.GELU(),
                nn.Linear(self.hidden_per_dim, self.layer_dim*2)  # outputs [mu_i, log_sigma_i]
        )
        # with torch.no_grad():
        #     for net in self.networks:
        #         net[-1].weight.data *= 0.1  # Small weights
        #         net[-1].bias.data[0] = 0.0   # mu starts at 0
        #         net[-1].bias.data[1] = 0.0   # log_sigma starts at 0 (sigma=1)
        

    def _instantaneous_gates_matrix(self) -> torch.Tensor:
        """Strictly lower triangular gates with temperature control"""
        g = stable_softplus(self.alpha_inst)
        # Clip to prevent extreme values
        g = torch.clamp(g, max=10.0)
        return torch.tril(g, diagonal=-1)

    def _delayed_gates_matrix(self) -> torch.Tensor:
        """Full positive gates with temperature control"""
        g = stable_softplus(self.beta_delay)
        # Clip to prevent extreme values
        g = torch.clamp(g, max=10.0)
        return g


    def forward_z_to_e(self,
                    z_t: torch.Tensor,
                    z_lagged: torch.Tensor,
                    context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute all dimensions in parallel using separate networks.
        
        Args:
            z_t: (batch_size, layer_dim)          
            z_lagged: (batch_size, layer_dim, window_size)
            context: (batch_size, context_dim)
        Returns:
            e_t: (batch_size, layer_dim)
            log_det: (batch_size,)
        """
        batch_size = z_t.shape[0]
        
        # Get gate matrices
        M = self._instantaneous_gates_matrix()     # (d, d) lower triangular
        Bm = self._delayed_gates_matrix()          # (d, d*w)
        
        # Flatten z_lagged: (B, d*w)
        z_lag_flat = z_lagged.reshape(batch_size, -1)
        x_in = torch.cat([z_t * 0, z_lag_flat, context], dim=-1)
        out = self.net(x_in)
        
        # Extract mu and log_sigma
        raw_mu = out[:, :self.layer_dim]           # (B, d)
        raw_log_sigma = out[:, self.layer_dim:]
        mu = raw_mu                      # no tanh
        # log_sigma = raw_log_sigma
        max_abs = 1.0
        log_sigma = max_abs * torch.tanh(raw_log_sigma / max_abs) #raw_log_sigma.clamp(-5.0, 5.0)
        sigma = torch.exp(log_sigma)
        # if torch.rand(()) < 0.001:  # sample occasionally
        #     print("raw_log_sigma", raw_log_sigma.min().item(), raw_log_sigma.max().item())
        #     print("log_sigma", log_sigma.min().item(), log_sigma.max().item())
        #     print("mu", raw_mu.min().item(), raw_mu.max().item())
        log_det = (-log_sigma).sum(-1)
        e = (z_t - mu) #/ sigma
        # mu = 10.0 * torch.tanh(raw_mu)
        # log_sigma = 2.0 * F.tanh(raw_log_sigma)       # hard floor
        # sigma = torch.exp(log_sigma)
        # log_det = (-log_sigma).sum(-1)
        # e = (z_t - mu) / sigma

        # clamp log det to prevent it from dominating
        # log_det_capped = torch.clamp(log_det, min=-10, max=10)
    
        return e, 0

    def log_prob(self,
                 z_t: torch.Tensor,
                 z_lagged: torch.Tensor,
                 context: torch.Tensor,
                 prior: NoisePrior) -> torch.Tensor:
        """
        Compute log q(z | z_lagged, context) for this layer.
        """
        e, log_det = self.forward_z_to_e(z_t=z_t, z_lagged=z_lagged, context=context)
        log_prior = prior.log_prob(e)
        
        # Check for NaN/Inf
        if torch.isnan(log_det).any() or torch.isinf(log_det).any():
            print(f"Warning: Invalid log_det values detected")
        if torch.isnan(log_prior).any() or torch.isinf(log_prior).any():
            print(f"Warning: Invalid log_prior values detected")
            
        return log_prior + 0 * log_det # to debug the log pe term, assume additive noise

    def gate_penalties(self) -> Dict[str, torch.Tensor]:
        """Compute L1 penalties on gate values for sparsity"""
        A = self._instantaneous_gates_matrix()
        Bm = self._delayed_gates_matrix()
        l1_inst = A.abs().sum()
        l1_delay = Bm.abs().sum()
        return {"l1_inst": l1_inst, "l1_delay": l1_delay}


class ConditionalFlow(nn.Module):
    """
    Stack of conditional MAF layers with optional permutations.

    For training log-likelihood, we operate in the z->e direction for each layer.
    Permutations keep determinant 1, so log_det unaffected.
    """
    def __init__(self, layer_dim: int, context_dim: int, window_size: int, 
                 n_layers: int = 4,
                 hidden: int = 512,
                 use_permutations: bool = True):
        super().__init__()
        self.layer_dim = layer_dim
        self.context_dim = context_dim
        self.n_layers = n_layers
        self.use_permutations = use_permutations

        self.layers = nn.ModuleList([
            SimpleGatedConditionalMAFLayer(layer_dim=layer_dim, window_size=window_size, context_dim=context_dim, hidden=hidden)
            for _ in range(n_layers)
        ])

        if use_permutations:
            perms = []
            invs = []
            for k in range(n_layers - 1):
                p = torch.randperm(layer_dim)
                inv = torch.empty_like(p)
                inv[p] = torch.arange(layer_dim)
                perms.append(p)
                invs.append(inv)
            self.register_buffer("perms", torch.stack(perms) if perms else torch.empty(0, layer_dim, dtype=torch.long))
            self.register_buffer("inv_perms", torch.stack(invs) if invs else torch.empty(0, layer_dim, dtype=torch.long))
        else:
            self.register_buffer("perms", torch.empty(0, layer_dim, dtype=torch.long))
            self.register_buffer("inv_perms", torch.empty(0, layer_dim, dtype=torch.long))

    def log_prob(self, z_t: torch.Tensor, z_lagged: torch.Tensor, context: torch.Tensor, prior: NoisePrior) -> torch.Tensor:
        """
        Compute log q(z | z_lagged, context) for stacked flow.
        For MAF-style stacking, we iteratively transform z -> e and treat e as new z for next layer
        (composing invertible transforms). Accumulate log-det and base logprob at final e.
        """
        # We'll implement "z -> e" per layer, and feed e as next layer's z.
        # Base prior always applied to final e (after all layers).
        h = z_t
        total_log_det = torch.zeros(z_t.shape[0], device=z_t.device, dtype=z_t.dtype)

        for idx, layer in enumerate(self.layers):
            e, log_det = layer.forward_z_to_e(h, z_lagged=z_lagged, context=context)
            total_log_det = total_log_det + log_det
            h = e
            # permutation between layers (except after last)
            if self.use_permutations and idx < self.n_layers - 1:
                p = self.perms[idx]
                h = h[:, p]

        # final h is base noise sample
        log_base = prior.log_prob(h)
        # log_base = 

        # print("log_base mean", log_base.mean().item(), "total_log_det mean", total_log_det.mean().item())
        return log_base + total_log_det 


    def gate_penalties(self) -> Dict[str, torch.Tensor]:
        l1_inst = 0.0
        l1_delay = 0.0
        for layer in self.layers:
            gp = layer.gate_penalties()
            l1_inst = l1_inst + gp["l1_inst"]
            l1_delay = l1_delay + gp["l1_delay"]
        return {"l1_inst": torch.as_tensor(l1_inst), "l1_delay": torch.as_tensor(l1_delay)}


# -----------------------------
# Whole model
# -----------------------------

@dataclass
class LossWeights:
    w_rec: float = 1.0   
    w_nq0: float = 1.0
    w_nq1: float = 1.0
    w_cmi: float = 1.0
    w_gate_inst: float = 0.0
    w_gate_delay: float = 0.0


class WholeModel(nn.Module):
    def __init__(self,
                 x_dim: int = 1000,
                 z_dim: int = 64,
                 u_dim: int = 128,
                 window_size: int = 1,  # Should match L from dataset
                 enc_width: int = 512,
                 enc_depth: int = 6,
                 dec_width: int = 512,
                 dec_depth: int = 6,
                 flow_layers: int = 4,
                 flow_hidden: int = 512,
                 prior_type: Literal["normal", "logistic"] = "normal",
                 prior_kwargs: Optional[dict] = None):
        super().__init__()
        prior_kwargs = prior_kwargs or {}

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.window_size = window_size

        # additive noise conditional entropy
        self.ce_xt_zt = ResidualMLP(in_dim=z_dim, out_dim=x_dim*2, width=enc_width, depth=enc_depth)
        self.ce_zt1_zt = ResidualMLP(in_dim=z_dim*self.window_size, out_dim=z_dim*2, width=enc_width, depth=enc_depth)
        self.ce_zt1_xt = ResidualMLP(in_dim=x_dim*self.window_size, out_dim=z_dim*2, width=enc_width, depth=enc_depth)

        # Recon net
        self.encoder = ResidualMLP(in_dim=x_dim, out_dim=z_dim, width=enc_width, depth=enc_depth)
        self.decoder = ResidualMLP(in_dim=z_dim, out_dim=x_dim, width=dec_width, depth=dec_depth)
        # self.encoder = InfoNCEEncoder(
        #     x_dim=x_dim,
        #     z_dim=z_dim,
        #     width=enc_width,
        #     depth=enc_depth,
        #     dropout=0.0,
        #     temperature=1.0,
        #     normalize=False,
        #     use_momentum_target=False,
        #     momentum=0.99,
        #     symmetric=True,
        # )
        # self.encoder = AugmentedInfoNCEEncoder(x_dim=x_dim, z_dim=z_dim, temperature=1.0, aug_noise=0.2)

        # Extra feature for q1
        self.x_to_u = ResidualMLP(in_dim=x_dim*window_size, out_dim=u_dim, width=enc_width // 2, depth=enc_depth)
        # self.x_to_u = nn.Sequential(
        #     nn.Linear(x_dim*window_size, enc_width // 2),
        #     nn.LayerNorm(enc_width // 2),
        #     nn.GELU(),
        #     nn.Linear(enc_width // 2, u_dim)
        # )
        # Priors
        self.prior = make_prior(prior_type, layer_dim=z_dim, **prior_kwargs)

        # Flows:
        # q0: models q(z_{t+1} | z_t, z_{t-1}, ..., z_{t-window_size+1})
        self.flow_q0 = ConditionalFlow(
            layer_dim=z_dim,
            window_size=window_size,
            context_dim=0,  # No extra context
            n_layers=flow_layers,
            hidden=flow_hidden,
            use_permutations=True
        )
        
        # q1: models q(z_{t+1} | z_t, z_{t-1}, ..., z_{t-window_size+1}, u_t)
        self.flow_q1 = ConditionalFlow(
            layer_dim=z_dim,
            window_size=window_size,
            context_dim=u_dim,
            n_layers=flow_layers,
            hidden=flow_hidden,
            use_permutations=False
        )

    def encode_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence of observations.
        
        Args:
            x_seq: (batch_size, seq_len, x_dim) or (seq_len, x_dim)
        Returns:
            z_seq: (batch_size, seq_len, z_dim) or (seq_len, z_dim)
        """
        original_shape = x_seq.shape
        needs_squeeze = x_seq.ndim == 2
        
        if needs_squeeze:
            x_seq = x_seq.unsqueeze(0)  # Add batch dim
        
        batch_size, seq_len, _ = x_seq.shape
        # Flatten to encode all at once
        x_flat = x_seq.reshape(batch_size * seq_len, self.x_dim)
        z_flat = self.encoder(x_flat)
        z_seq = z_flat.reshape(batch_size, seq_len, self.z_dim)
        
        if needs_squeeze:
            z_seq = z_seq.squeeze(0)  # Remove batch dim
        
        return z_seq

    def log_q0(self, z_t: torch.Tensor, z_lagged: torch.Tensor) -> torch.Tensor:
        """
        Compute log q0(z_{t+1} | z_lagged).
        
        Args:
            z_t: (batch_size, z_dim)
            z_lagged: (batch_size, z_dim, window_size)
        Returns:
            log_prob: (batch_size,)
        """
        ctx0 = z_t.new_zeros((z_t.shape[0], 0))
        return self.flow_q0.log_prob(z_t=z_t, z_lagged=z_lagged, context=ctx0, prior=self.prior)

    def log_q1(self, z_t: torch.Tensor, z_lagged: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Compute log q1(z_{t+1} | z_lagged, u_t).
        
        Args:
            z_t: (batch_size, z_dim)
            z_lagged: (batch_size, z_dim, window_size)
            context: (batch_size, u_dim)
        Returns:
            log_prob: (batch_size,)
        """
        z_lagged0 = torch.zeros_like(z_lagged)
        return self.flow_q1.log_prob(z_t=z_t, z_lagged=z_lagged0, context=context, prior=self.prior)

    # def forward(self,
    #             x_lags: torch.Tensor,
    #             x_curr: torch.Tensor,
    #             weights: LossWeights = LossWeights(),
    #             mode: str = "enc") -> Dict[str, torch.Tensor]:

    #     batch_size = x_lags.shape[0]
    #     L = x_lags.shape[1]
    #     if L != self.window_size:
    #         raise ValueError(f"x_lags has length {L} but model expects window_size={self.window_size}")

    #     # ---- inputs
    #     x_t = x_curr.squeeze(1)  # (B, x_dim)

    #     # ---- encode lag window
    #     z_lags = self.encode_sequence(x_lags)  # (B, L, z_dim)

    #     # ---- encode current
    #     z_t = self.encoder(x_t)  # (B, z_dim)

    #     # =========================
    #     # Gradient control switches
    #     # =========================
    #     if mode == "pred":
    #         # Predictors update: don't let gradients flow into encoder through z_t or z_lags
    #         z_t_used = z_t.detach()
    #         z_lags_used = z_lags.detach()
    #     elif mode == "enc":
    #         # Encoder update:
    #         # - baseline path uses stop-grad inputs so encoder can't sabotage baseline
    #         # - strong path keeps grads (so encoder learns to remove extra predictability)
    #         z_t_used = z_t  # allow grads for x_t <-> z_t term and strong path (if it uses z_t)
    #         z_lags_used = z_lags.detach()  # baseline predictor input stop-grad
    #     else:
    #         raise ValueError(f"Unknown mode={mode}. Use 'pred' or 'enc'.")

    #     # # =========================
    #     # # 1) "Info" term: x_t from z_t
    #     # # =========================
    #     # mu_xt_zt, raw_logsigma_xt_zt = self.ce_xt_zt(z_t_used)[:,:self.x_dim], self.ce_xt_zt(z_t_used)[:,self.x_dim:]  # (B, x_dim), (B, x_dim)
    #     # logsigma_xt_zt = 5.0 * torch.tanh(raw_logsigma_xt_zt / 5.0)
    #     # L_ce_xt_zt = gaussian_nll_diag(x_t, mu_xt_zt, logsigma_xt_zt)

    #     # # =========================
    #     # # 2) Baseline predictor: z_t from past z_lags
    #     # # =========================
    #     # # IMPORTANT: baseline predicts in z-space, target is z_t (not x_t)
    #     # mu_zt_from_zlags, raw_logsigma_zt_from_zlags = self.ce_zt1_zt(z_lags_used.reshape(batch_size, -1))[:, :self.z_dim], self.ce_zt1_zt(z_lags_used.reshape(batch_size, -1))[:, self.z_dim:]  # (B, z_dim), (B, z_dim)
    #     # logsigma_zt_from_zlags = 5.0 * torch.tanh(raw_logsigma_zt_from_zlags / 5.0)
    #     # L_ce_zt_from_zlags = gaussian_nll_diag(z_t.detach() if mode == "pred" else z_t,  # target can be non-detached in enc mode
    #     #                                     mu_zt_from_zlags,
    #     #                                     logsigma_zt_from_zlags)

    #     # # =========================
    #     # # 3) Strong predictor: z_t from x_lags (or include x_t, your choice)
    #     # # =========================
    #     # # If your intent is "does x_t add extra predictability beyond z_lags?", then
    #     # # make strong predictor take (x_t, z_lags). But with your current dataset you used x_t only.
    #     # # Here I keep your current signature but FIX target to z_t.
    #     # mu_zt_from_xt, raw_logsigma_zt_from_xt = self.ce_zt1_xt(x_t.reshape(batch_size, -1))[:, :self.z_dim], self.ce_zt1_xt(x_t.reshape(batch_size, -1))[:, self.z_dim:]  # (B, z_dim), (B, z_dim)
    #     # logsigma_zt_from_xt = 5.0 * torch.tanh(raw_logsigma_zt_from_xt / 5.0)
    #     # L_ce_zt_from_xt = gaussian_nll_diag(z_t,  # allow grads to encoder through z_t in enc mode
    #     #                                     mu_zt_from_xt,
    #     #                                     logsigma_zt_from_xt)

    #     # =========================
    #     # Combine losses (example)
    #     # =========================
    #     # out: Dict[str, torch.Tensor] = {}
    #     # out["L_ce_xt_zt"] = L_ce_xt_zt
    #     # out["L_ce_zt_from_zlags"] = L_ce_zt_from_zlags
    #     # out["L_ce_zt_from_xt"] = L_ce_zt_from_xt
    #     # L_cmi = L_ce_zt_from_zlags - L_ce_zt_from_xt
    #     out = {
    #         "L_rec": L_ce_xt_zt,
    #         "L_nq0": L_ce_zt_from_zlags,
    #         "L_nq1": L_ce_zt_from_xt,
    #         "L_cmi": L_cmi,
    #         "L_gate_inst": 0.,
    #         "L_gate_delay": 0.,
    #         # "infonce_acc": inf_stats["acc_inbatch"].detach(),
    #         # "logq0_mean": logq0.mean().detach(),
    #         # "logq1_mean": logq1.mean().detach(),
    #         "logq0_mean": 0.,
    #         "logq1_mean": 0.,
    #     }

    #     # Predictor-only objective (train predictors to be strong)
    #     out["L_pred"] = (
    #         weights.w_rec * L_ce_xt_zt +
    #         weights.w_nq0  * L_ce_zt_from_zlags +
    #         weights.w_nq1 * L_ce_zt_from_xt
    #     )

    #     # Encoder objective:
    #     # - encourage z to explain x: minimize L_ce_xt_zt
    #     # - minimize predictability gap: (baseline - strong) should go DOWN, so minimize (L_base - L_strong)
    #     # NOTE: baseline input is stop-grad in enc mode, so encoder can't sabotage baseline.
    #     out["L_enc"] = (
    #         weights.w_rec * L_ce_xt_zt +
    #         weights.w_cmi * (L_cmi)
    #     )

    #     # Total depends on mode
    #     out["L_total"] = out["L_pred"] if mode == "pred" else out["L_enc"]

    #     return out


    def forward(self,
                x_lags: torch.Tensor,
                x_curr: torch.Tensor,
                weights: LossWeights = LossWeights(),
                mode: str = "enc") -> Dict[str, torch.Tensor]:
        """
        Forward pass matching LagWindowDataset output.
        
        Args:
            x_lags: (batch_size, L, x_dim) - historical observations [x_{t-L}, ..., x_{t-1}]
            x_curr: (batch_size, 1, x_dim) - current observation [x_t]
            weights: loss weights
            
        Returns:
            Dictionary with losses and diagnostics
        """
        batch_size = x_lags.shape[0]
        L = x_lags.shape[1]
        
        # Verify window size matches
        if L != self.window_size:
            raise ValueError(f"x_lags has length {L} but model expects window_size={self.window_size}")
        
        # Encode the lag window: x_{t-L}, ..., x_{t-1}
        # x_lags: (B, L, n) -> z_lags: (B, L, z_dim)
        z_lags = self.encode_sequence(x_lags)  # (B, L, z_dim)
        z_lags_stand = batch_standardize(z_lags)
        
        # Encode current: x_t
        # x_curr: (B, 1, n) -> z_curr: (B, 1, z_dim)
        x_t = x_curr.squeeze(1)  # (B, n)
        z_t = self.encoder(x_t)  # (B, z_dim)

        # mu_xt_zt, raw_logsigma_xt_zt = self.ce_xt_zt(z_t)
        # logsigma_xt_zt = 5.0 * torch.tanh(raw_logsigma_xt_zt / 5.0)
        # sigma_xt_zt = torch.exp(logsigma_xt_zt)
        # E_xt_zt = (x_t - mu_xt_zt) / sigma_xt_zt
        # L_ce_xt_zt = (E_xt_zt**2 / sigma_xt_zt).sum(dim=-1).mean() + sigma_xt_zt.prod(dim=-1).abs().mean()

        # mu_zt1_zt, raw_logsigma_zt1_zt = self.ce_zt1_zt(z_lags)
        # logsigma_zt1_zt = 5.0 * torch.tanh(raw_logsigma_zt1_zt / 5.0)
        # sigma_zt1_zt = torch.exp(logsigma_zt1_zt)
        # E_zt1_zt = (x_t - mu_zt1_zt) / sigma_zt1_zt
        # L_ce_zt1_zt = (E_zt1_zt**2 / sigma_zt1_zt).sum(dim=-1).mean() + sigma_zt1_zt.prod(dim=-1).abs().mean()

        # mu_zt1_xt, raw_logsigma_zt1_xt = self.ce_zt1_xt(x_t)
        # logsigma_zt1_xt = 5.0 * torch.tanh(raw_logsigma_zt1_xt / 5.0)
        # sigma_zt1_xt = torch.exp(logsigma_zt1_xt)
        # E_zt1_xt = (x_t - mu_zt1_xt) / sigma_zt1_xt
        # L_ce_zt1_xt = (E_zt1_xt**2 / sigma_zt1_xt).sum(dim=-1).mean() + sigma_zt1_xt.prod(dim=-1).abs().mean()

        z_t_stand = batch_standardize(z_t)

        # print(f"z_t has NaN: {torch.isnan(z_t).any()}")
        # print(f"z_t has Inf: {torch.isinf(z_t).any()}")
        # print(f"z_t stats: mean={z_t.mean():.4f}, std={z_t.std():.4f}, min={z_t.min():.4f}, max={z_t.max():.4f}")
        
        # Prepare z_lagged for flow: need (B, z_dim, window_size)
        # z_lags is (B, L, z_dim) -> transpose to (B, z_dim, L)
        z_lags_stand = z_lags_stand.transpose(1, 2)  # (B, z_dim, L)
        
        # Reconstruction loss: reconstruct x_{t-1} from z_{t-1}
        # (We reconstruct the most recent historical point)
        # z_t_minus_1 = z_lags[:, -1, :]  # (B, z_dim) - last element of lag window
        # x_t_minus_1 = x_lags[:, -1, :]  # (B, n) - last element of lag window
        # L_rec, inf_stats = self.encoder.infonce_loss(x_t)
        x_hat = self.decoder(z_t_stand)

        L_rec = F.mse_loss(x_hat, x_t, reduction="mean")
        # L_rec = (0.1 - z_t.var()).relu()
        
        # q0 loglik: log q0(z_t | z_{t-L}, ..., z_{t-1})
        logq0 = self.log_q0(z_t=z_t, z_lagged=z_lags)
        L_nq0 = -logq0.mean()
        
        # q1 loglik: log q1(z_t | z_{t-L}, ..., z_{t-1}, u_{t-1})
        u_t = self.x_to_u(x_lags.reshape(batch_size, -1))
        u_t_stand = batch_standardize(u_t)
        logq1 = self.log_q1(z_t=z_t, z_lagged=z_lags, context=u_t)
        L_nq1 = -logq1.mean()
        
        # CMI penalty: minimize information gain from u
        L_cmi = torch.relu(logq1 - logq0).mean()
        
        # Gate sparsity penalties
        gp0 = self.flow_q0.gate_penalties()
        gp1 = self.flow_q1.gate_penalties()
        L_gate_inst = gp0["l1_inst"] + gp1["l1_inst"]
        L_gate_delay = gp0["l1_delay"] + gp1["l1_delay"]
        # L_recon = L_ce_xt_zt
        # L_nq0 = (L_ce_zt1_zt - L_ce_zt1_xt).relu()
        # L_nq1 = L_ce_zt1_zt
        # L_cmi = L_ce_zt1_xt
        L_gate_delay = 0.
        L_gate_inst = 0.
        
        L_total = (
            weights.w_rec * L_rec
            + weights.w_nq0 * L_nq0
            + weights.w_nq1 * L_nq1
            + weights.w_cmi * L_cmi
            + weights.w_gate_inst * L_gate_inst
            + weights.w_gate_delay * L_gate_delay
        )
        
        return {
            "L_rec": L_rec,
            "L_nq0": L_nq0,
            "L_nq1": L_nq1,
            "L_cmi": L_cmi,
            "L_gate_inst": L_gate_inst,
            "L_gate_delay": L_gate_delay,
            "L_total": L_total,
            # "infonce_acc": inf_stats["acc_inbatch"].detach(),
            # "logq0_mean": logq0.mean().detach(),
            # "logq1_mean": logq1.mean().detach(),
            "logq0_mean": 0.,
            "logq1_mean": 0.,
        }
    
