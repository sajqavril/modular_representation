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


# -----------------------------
# Utilities
# -----------------------------

def stable_softplus(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.softplus(x) + eps


def huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    # elementwise Huber
    absx = x.abs()
    quad = torch.where(absx <= delta, 0.5 * x * x, delta * (absx - 0.5 * delta))
    return quad


# -----------------------------
# Noise prior p_E(e)
# -----------------------------

class NoisePrior(nn.Module):
    """Base class: factorized noise prior p_E(e)=∏_i p(e_i)."""
    def log_prob(self, e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e: (B, d)
        Returns:
            log p_E(e): (B,)
        """
        raise NotImplementedError


class StandardNormalPrior(NoisePrior):
    def log_prob(self, e: torch.Tensor) -> torch.Tensor:
        # sum over dims
        return (-0.5 * (e ** 2) - 0.5 * math.log(2 * math.pi)).sum(dim=-1)


class StandardLogisticPrior(NoisePrior):
    def log_prob(self, e: torch.Tensor) -> torch.Tensor:
        # Logistic(0,1): log pdf = -e - 2*softplus(-e)
        # stable form:
        return (-e - 2.0 * F.softplus(-e)).sum(dim=-1)


class StudentTPrior(NoisePrior):
    """
    Factorized Student-t with df=nu, loc=0, scale=1.
    nu can be fixed or learnable.
    """
    def __init__(self, nu: float = 5.0, learnable: bool = False):
        super().__init__()
        if learnable:
            self.log_nu = nn.Parameter(torch.tensor(float(math.log(nu))))
        else:
            self.register_buffer("log_nu", torch.tensor(float(math.log(nu))))

    def log_prob(self, e: torch.Tensor) -> torch.Tensor:
        # log pdf for Student-t df=nu, scale=1:
        # log Γ((ν+1)/2) - log(√(νπ) Γ(ν/2)) - (ν+1)/2 * log(1 + e^2/ν)
        nu = self.log_nu.exp()
        c = torch.lgamma((nu + 1.0) / 2.0) - torch.lgamma(nu / 2.0) - 0.5 * (torch.log(nu) + math.log(math.pi))
        lp = c - 0.5 * (nu + 1.0) * torch.log1p((e * e) / nu)
        return lp.sum(dim=-1)


class PerDimGaussianMixturePrior(NoisePrior):
    """
    Factorized per-dimension mixture of Gaussians:
    p(e_i)=∑_k π_{ik} N(e_i; μ_{ik}, σ_{ik}^2)
    This keeps independence across dimensions but flexible marginals.
    """
    def __init__(self, d: int, K: int = 4):
        super().__init__()
        self.d = d
        self.K = K
        # logits for mixing weights (d, K)
        self.logits = nn.Parameter(torch.zeros(d, K))
        self.mu = nn.Parameter(torch.zeros(d, K))
        self.log_sigma = nn.Parameter(torch.zeros(d, K))  # unconstrained

    def log_prob(self, e: torch.Tensor) -> torch.Tensor:
        # e: (B, d)
        B, d = e.shape
        assert d == self.d
        # Expand to (B, d, K)
        e_ = e.unsqueeze(-1)
        mu = self.mu.unsqueeze(0)               # (1, d, K)
        sigma = stable_softplus(self.log_sigma).unsqueeze(0)  # (1, d, K)
        # log N
        log_norm = -0.5 * math.log(2 * math.pi) - torch.log(sigma) - 0.5 * ((e_ - mu) / sigma) ** 2
        # mix log probs
        log_pi = F.log_softmax(self.logits, dim=-1).unsqueeze(0)  # (1, d, K)
        # logsumexp over K, sum over d
        log_p_dim = torch.logsumexp(log_pi + log_norm, dim=-1)  # (B, d)
        return log_p_dim.sum(dim=-1)


def make_prior(prior_type: Literal["normal", "logistic", "studentt", "mog"],
               d: int,
               **kwargs) -> NoisePrior:
    if prior_type == "normal":
        return StandardNormalPrior()
    if prior_type == "logistic":
        return StandardLogisticPrior()
    if prior_type == "studentt":
        return StudentTPrior(**kwargs)
    if prior_type == "mog":
        return PerDimGaussianMixturePrior(d=d, **kwargs)
    raise ValueError(f"Unknown prior_type={prior_type}")


# -----------------------------
# Encoder / Decoder (Recon net)
# -----------------------------

class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 512, depth: int = 6,
                 dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        self.dropout = dropout

        for _ in range(depth):
            self.norms1.append(nn.LayerNorm(width))
            self.norms2.append(nn.LayerNorm(width))
            self.fcs1.append(nn.Linear(width, width))
            self.fcs2.append(nn.Linear(width, width))

        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for n1, n2, fc1, fc2 in zip(self.norms1, self.norms2, self.fcs1, self.fcs2):
            r = n1(h)
            r = F.gelu(fc1(r))
            r = n2(r)
            r = fc2(r)
            if self.dropout > 0:
                r = F.dropout(r, p=self.dropout, training=self.training)
            h = h + r
        h = self.out_norm(h)
        return self.out_proj(h)


# -----------------------------
# Gated conditional autoregressive flow (MAF-style layer)
# -----------------------------

class GatedConditionalMAFLayer(nn.Module):
    """
    One conditional MAF layer:
      e_i = (z_i - mu_i(context, z_<i)) / sigma_i(context, z_<i)

    We implement mu/log_sigma via a masked MLP that enforces dependence only on z_<i.
    Additionally, we apply learnable gates to:
      - instantaneous inputs: z_<i (lower triangular gate matrix A_inst)
      - delayed inputs: z_t (full gate matrix B_delay)

    Notes:
      - This layer computes log q(z|context) in "forward" direction (z -> e).
      - For simplicity and clarity, we implement masking via explicit elementwise gating
        + a simple MADE-like mask on the first linear layer for z inputs.
      - You can replace the internal masked net with a more standard MADE implementation later.
    """

    def __init__(self, d: int, context_dim: int, hidden: int = 512):
        super().__init__()
        self.d = d
        self.context_dim = context_dim
        self.hidden = hidden

        # Learnable gates:
        # Instantaneous: strictly lower triangular (d, d) but we only use lower part.
        self.alpha_inst = nn.Parameter(torch.zeros(d, d))   # raw parameters
        # Delayed: full (d, d_delay) where z_t dimension assumed == d.
        # If z_t dim differs from z_{t+1} dim, set delay_dim separately (see below).
        self.delay_dim = d
        self.beta_delay = nn.Parameter(torch.zeros(d, self.delay_dim))

        # A simple masked network:
        # Inputs: [gated_z (d), gated_delay (delay_dim), context (context_dim)]
        # Output: [mu (d), log_sigma (d)]
        in_dim = d + self.delay_dim + context_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, 2 * d)

        # Create a fixed autoregressive mask for the z part into fc1:
        # output dims depend on hidden units; simplest: mask z inputs so each hidden unit sees only prefix.
        # We'll implement a coarse mask: split hidden into d groups.
        self.register_buffer("z_to_hidden_mask", self._make_z_to_hidden_mask(d, hidden))

    @staticmethod
    def _make_z_to_hidden_mask(d: int, hidden: int) -> torch.Tensor:
        """
        Create mask for fc1 weights corresponding to z inputs only: (hidden, d).
        Hidden units are divided into d groups; group i sees z_<=(i-1).
        """
        mask = torch.zeros(hidden, d)
        # assign each hidden unit a "degree" in {0..d-1}
        degrees = torch.arange(hidden) % d  # 0..d-1
        for h in range(hidden):
            deg = int(degrees[h].item())
            # allow inputs 0..deg-1 (i.e., z_<i style)
            if deg > 0:
                mask[h, :deg] = 1.0
        return mask

    def _instantaneous_gates_matrix(self) -> torch.Tensor:
        # strictly lower triangular gates, positive via softplus
        g = stable_softplus(self.alpha_inst)
        return torch.tril(g, diagonal=-1)

    def _delayed_gates_matrix(self) -> torch.Tensor:
        # full positive gates
        return stable_softplus(self.beta_delay)

    def forward_z_to_e(self,
                       z: torch.Tensor,
                       z_delay: torch.Tensor,
                       context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, d)          -> here z is z_{t+1}
            z_delay: (B, delay_dim) -> here z_delay is z_t (assumed same dim; can adapt)
            context: (B, context_dim) -> extra conditioning (e.g., u_t)
        Returns:
            e: (B, d)
            log_det: (B,)   log |det de/dz|
        """
        B, d = z.shape
        assert d == self.d
        assert z_delay.shape[0] == B
        assert context.shape[0] == B

        A = self._instantaneous_gates_matrix()   # (d, d) lower triangular
        Bm = self._delayed_gates_matrix()        # (d, delay_dim)

        # Build gated instantaneous inputs for each dimension i:
        # For a MADE net, we'd feed z directly and rely on masks.
        # Here we also apply an explicit gating transform z_g = z @ A^T  (uses only lower triangle).
        # z_g[i] = sum_{j<i} a_{ij} z_j  (instantaneous influence summary)
        z_inst = z @ A.t()  # (B, d)

        # Build gated delayed summary: z_delay_g[i] = sum_k b_{ik} z_delay[k]
        z_delay_g = z_delay @ Bm.t()  # (B, d)

        # Input to the network
        x_in = torch.cat([z_inst, z_delay_g, context], dim=-1)  # (B, 2d + context_dim)

        # Apply the MADE-like mask on fc1 weights for the z_inst portion only
        # We'll manually mask the first d columns of fc1.weight during forward.
        w = self.fc1.weight
        b = self.fc1.bias

        # Split fc1 weight into [z_inst (d), z_delay_g (d), context]
        w_z = w[:, :d]                      # (hidden, d)
        w_rest = w[:, d:]                   # (hidden, d + context_dim)

        # mask z part
        w_z_masked = w_z * self.z_to_hidden_mask

        h = F.linear(x_in, torch.cat([w_z_masked, w_rest], dim=1), b)
        h = F.gelu(h)
        h = F.gelu(self.fc2(h))
        out = self.fc_out(h)  # (B, 2d)
        mu, log_sigma = out[:, :d], out[:, d:]
        sigma = stable_softplus(log_sigma)

        e = (z - mu) / sigma
        log_det = (-torch.log(sigma)).sum(dim=-1)  # sum_i log |∂e_i/∂z_i| = -sum log sigma_i
        return e, log_det

    def log_prob(self,
                 z: torch.Tensor,
                 z_delay: torch.Tensor,
                 context: torch.Tensor,
                 prior: NoisePrior) -> torch.Tensor:
        """
        Compute log q(z | z_delay, context) for this layer only (one flow layer).
        For multiple layers, you'll wrap them and accumulate.
        """
        e, log_det = self.forward_z_to_e(z=z, z_delay=z_delay, context=context)
        return prior.log_prob(e) + log_det

    def gate_penalties(self) -> Dict[str, torch.Tensor]:
        A = self._instantaneous_gates_matrix()
        Bm = self._delayed_gates_matrix()
        # L1 sparsity on gate values
        l1_inst = A.abs().sum()
        l1_delay = Bm.abs().sum()
        return {"l1_inst": l1_inst, "l1_delay": l1_delay}


class ConditionalFlow(nn.Module):
    """
    Stack of conditional MAF layers with optional permutations.

    For training log-likelihood, we operate in the z->e direction for each layer.
    Permutations keep determinant 1, so log_det unaffected.
    """
    def __init__(self, d: int, context_dim: int,
                 n_layers: int = 4,
                 hidden: int = 512,
                 use_permutations: bool = True):
        super().__init__()
        self.d = d
        self.context_dim = context_dim
        self.n_layers = n_layers
        self.use_permutations = use_permutations

        self.layers = nn.ModuleList([
            GatedConditionalMAFLayer(d=d, context_dim=context_dim, hidden=hidden)
            for _ in range(n_layers)
        ])

        if use_permutations:
            perms = []
            invs = []
            for k in range(n_layers - 1):
                p = torch.randperm(d)
                inv = torch.empty_like(p)
                inv[p] = torch.arange(d)
                perms.append(p)
                invs.append(inv)
            self.register_buffer("perms", torch.stack(perms) if perms else torch.empty(0, d, dtype=torch.long))
            self.register_buffer("inv_perms", torch.stack(invs) if invs else torch.empty(0, d, dtype=torch.long))
        else:
            self.register_buffer("perms", torch.empty(0, d, dtype=torch.long))
            self.register_buffer("inv_perms", torch.empty(0, d, dtype=torch.long))

    def log_prob(self, z: torch.Tensor, z_delay: torch.Tensor, context: torch.Tensor, prior: NoisePrior) -> torch.Tensor:
        """
        Compute log q(z | z_delay, context) for stacked flow.
        For MAF-style stacking, we iteratively transform z -> e and treat e as new z for next layer
        (composing invertible transforms). Accumulate log-det and base logprob at final e.
        """
        # We'll implement "z -> e" per layer, and feed e as next layer's z.
        # Base prior always applied to final e (after all layers).
        # But each layer adds its own log_det. This is standard flow composition.
        h = z
        total_log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        for idx, layer in enumerate(self.layers):
            e, log_det = layer.forward_z_to_e(h, z_delay=z_delay, context=context)
            total_log_det = total_log_det + log_det
            h = e
            # permutation between layers (except after last)
            if self.use_permutations and idx < self.n_layers - 1:
                p = self.perms[idx]
                h = h[:, p]

        # final h is base noise sample
        log_base = prior.log_prob(h)
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
    w_noise: float = 1.0
    w_cmi: float = 1.0
    w_gate_inst: float = 0.0
    w_gate_delay: float = 0.0
    # optional: use huber for recon
    recon_huber_delta: Optional[float] = None


class WholeModel(nn.Module):
    def __init__(self,
                 x_dim: int = 1000,
                 z_dim: int = 64,
                 u_dim: int = 128,
                 enc_width: int = 512,
                 enc_depth: int = 6,
                 dec_width: int = 512,
                 dec_depth: int = 6,
                 flow_layers: int = 4,
                 flow_hidden: int = 512,
                 prior_type: Literal["normal", "logistic", "studentt", "mog"] = "logistic",
                 prior_kwargs: Optional[dict] = None):
        super().__init__()
        prior_kwargs = prior_kwargs or {}

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.u_dim = u_dim

        # Recon net
        self.encoder = ResidualMLP(in_dim=x_dim, out_dim=z_dim, width=enc_width, depth=enc_depth)
        self.decoder = ResidualMLP(in_dim=z_dim, out_dim=x_dim, width=dec_width, depth=dec_depth)

        # Extra feature for q1
        self.x_to_u = ResidualMLP(in_dim=x_dim, out_dim=u_dim, width=enc_width, depth=max(2, enc_depth // 2))

        # Priors
        self.prior = make_prior(prior_type, d=z_dim, **prior_kwargs)

        # Flows:
        # q0 context is empty additional (we treat context_dim=0) but we still pass something (zeros)
        self.flow_q0 = ConditionalFlow(d=z_dim, context_dim=0, n_layers=flow_layers, hidden=flow_hidden)
        # q1 context uses u_t
        self.flow_q1 = ConditionalFlow(d=z_dim, context_dim=u_dim, n_layers=flow_layers, hidden=flow_hidden)

    def encode_pair(self, x_t: torch.Tensor, x_tp1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_t = self.encoder(x_t)
        z_tp1 = self.encoder(x_tp1)
        return z_t, z_tp1

    def log_q0(self, z_tp1: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        # context_dim=0 => pass empty tensor
        ctx0 = z_tp1.new_zeros((z_tp1.shape[0], 0))
        return self.flow_q0.log_prob(z=z_tp1, z_delay=z_t, context=ctx0, prior=self.prior)

    def log_q1(self, z_tp1: torch.Tensor, z_t: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        return self.flow_q1.log_prob(z=z_tp1, z_delay=z_t, context=u_t, prior=self.prior)

    def forward(self,
                x_t: torch.Tensor,
                x_tp1: torch.Tensor,
                weights: LossWeights) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
          - losses: L_rec, L_noise, L_cmi, L_gate_inst, L_gate_delay, L_total
          - diagnostics: logq0_mean, logq1_mean
        """
        z_t, z_tp1 = self.encode_pair(x_t, x_tp1)

        # Reconstruction
        x_hat = self.decoder(z_t)
        if weights.recon_huber_delta is None:
            L_rec = F.mse_loss(x_hat, x_t, reduction="mean")
        else:
            L_rec = huber(x_hat - x_t, delta=weights.recon_huber_delta).mean()

        # q0 loglik (noise/transition likelihood)
        logq0 = self.log_q0(z_tp1=z_tp1, z_t=z_t)
        L_noise = -logq0.mean()

        # q1 loglik (extra access to x via u)
        u_t = self.x_to_u(x_t)
        logq1 = self.log_q1(z_tp1=z_tp1, z_t=z_t, u_t=u_t)

        # Two-conditionals CMI penalty
        # Minimizing E[log q1 - log q0] pushes q1 advantage to ~0.
        L_cmi = (logq1 - logq0).mean()

        # Gate sparsity penalties (sum over layers)
        gp0 = self.flow_q0.gate_penalties()
        gp1 = self.flow_q1.gate_penalties()

        # Usually you want structure to live in q0; sharing gates would be better,
        # but here we simply penalize both; you can later tie them together.
        L_gate_inst = gp0["l1_inst"] + gp1["l1_inst"]
        L_gate_delay = gp0["l1_delay"] + gp1["l1_delay"]

        L_total = (
            weights.w_rec * L_rec
            + weights.w_noise * L_noise
            + weights.w_cmi * L_cmi
            + weights.w_gate_inst * L_gate_inst
            + weights.w_gate_delay * L_gate_delay
        )

        return {
            "L_rec": L_rec,
            "L_noise": L_noise,
            "L_cmi": L_cmi,
            "L_gate_inst": L_gate_inst,
            "L_gate_delay": L_gate_delay,
            "L_total": L_total,
            "logq0_mean": logq0.mean().detach(),
            "logq1_mean": logq1.mean().detach(),
        }


# -----------------------------
# Jacobian helpers (for later constraints)
# -----------------------------

def jvp_of_flow_sample(model: WholeModel,
                       z_t: torch.Tensor,
                       e: torch.Tensor,
                       v: torch.Tensor) -> torch.Tensor:
    """
    Placeholder illustrating how you'd compute JVPs if you implement a sampling path
    z_{t+1} = T^{-1}(e; z_t).

    Note: The above flow implementation is log_prob-only (z->e direction).
    To compute Jacobians of z_{t+1} wrt z_t, you'd implement inverse/sampling for the flow,
    then use torch.autograd.functional.jvp or manual JVP/VJP.

    This function is a stub to show where it would hook in.
    """
    raise NotImplementedError("Implement inverse sampling for the flow to use JVP here.")


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    B = 32
    x_dim = 1000
    z_dim = 32

    model = WholeModel(
        x_dim=x_dim,
        z_dim=z_dim,
        u_dim=64,
        enc_width=512,
        enc_depth=6,
        dec_width=512,
        dec_depth=6,
        flow_layers=3,
        flow_hidden=512,
        prior_type="logistic",
    )

    x_t = torch.randn(B, x_dim)
    x_tp1 = torch.randn(B, x_dim)

    weights = LossWeights(
        w_rec=1.0,
        w_noise=1.0,
        w_cmi=0.1,
        w_gate_inst=1e-4,
        w_gate_delay=1e-4,
        recon_huber_delta=None,
    )

    out = model(x_t, x_tp1, weights)
    print({k: float(v) for k, v in out.items() if k.startswith("L_")})
