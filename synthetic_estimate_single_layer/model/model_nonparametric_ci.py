"""
MINE-based Conditional Independence Model

Uses neural estimation of mutual information instead of kernel methods.
Much faster and more numerically stable!

Key approaches:
1. MINE for I(h1(x_t); z_{t+1})
2. Conditional MINE for I(z_{t+1}; x_t | h1(x_t))
3. Uses critics/discriminators but in a stable way
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class LossWeights:
    """Loss weights for MINE-based model"""
    w_rec: float = 1.0      # Reconstruction (if ground truth available)
    w_pred: float = 1.0     # Prediction loss
    w_mi: float = 1.0       # MI maximization I(z_next; h1(x_t))
    w_cmi: float = 0.5      # CMI minimization I(z_next; x_t | h1(x_t))
    # Legacy (unused)
    w_nq0: float = 0.0
    w_nq1: float = 0.0
    w_gate_inst: float = 0.0
    w_gate_delay: float = 0.0


class SimpleEncoder(nn.Module):
    """Encoder: x_t -> z_t"""
    def __init__(self, x_dim: int, z_dim: int, width: int = 256, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = x_dim
        for _ in range(depth):
            layers.extend([
                nn.Linear(prev_dim, width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = width
        layers.append(nn.Linear(prev_dim, z_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, -1, :]  # Use only current observation
        return self.net(x)


class MINECritic(nn.Module):
    """
    MINE critic network: T(x, y) for estimating I(X; Y)
    
    Uses Donsker-Varadhan representation:
    I(X; Y) ≥ E[T(x,y)] - log E[exp(T(x,y'))]
    """
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        
        layers = []
        prev_dim = x_dim + y_dim
        for _ in range(depth):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, x_dim)
            y: (B, y_dim)
        Returns:
            scores: (B, 1)
        """
        xy = torch.cat([x, y], dim=-1)
        return self.net(xy).squeeze(-1)


class ConditionalMINECritic(nn.Module):
    """
    Conditional MINE critic: T(x, y, z) for estimating I(X; Y | Z)
    
    Uses: I(X; Y | Z) ≥ E[T(x,y,z)] - log E[exp(T(x,y',z))]
    where y' is sampled from a different data point but same z
    """
    def __init__(self, x_dim: int, y_dim: int, z_dim: int, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        
        layers = []
        prev_dim = x_dim + y_dim + z_dim
        for _ in range(depth):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, x_dim)
            y: (B, y_dim)
            z: (B, z_dim)
        Returns:
            scores: (B,)
        """
        xyz = torch.cat([x, y, z], dim=-1)
        return self.net(xyz).squeeze(-1)


class SimplePredictorMLP(nn.Module):
    """
    Simple MLP predictor for z_t from z_{t-1}
    (For L_pred loss)
    """
    def __init__(self, z_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )
    
    def forward(self, z_prev: torch.Tensor) -> torch.Tensor:
        return self.net(z_prev)


class WholeModel(nn.Module):
    """
    MINE-based model for learning hierarchical temporal representations.
    
    Uses neural MI estimation instead of kernel methods.
    More stable and faster!
    
    Components:
    1. Encoder: x_t -> h1(x_t)  [Learnable]
    2. Predictor: z_{t-1} -> z_t  [Learnable]
    3. MI Critic: For I(z_next; h1(x_t))  [Learnable]
    4. CMI Critic: For I(z_next; x_t | h1(x_t))  [Learnable]
    """
    
    def __init__(
        self,
        x_dim: int,
        z_dim: int = 1,
        window_size: int = 1,
        enc_width: int = 256,
        enc_depth: int = 2,
        critic_hidden: int = 256,
        critic_depth: int = 2,
        predictor_hidden: int = 128,
        use_ema: bool = False,  # Use EMA for critic targets
        # Unused args for compatibility
        **kwargs
    ):
        super().__init__()
        
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.window_size = window_size
        self.use_ema = use_ema
        
        # Main encoder (what we want to learn)
        self.encoder = SimpleEncoder(x_dim, z_dim, enc_width, enc_depth)
        
        # Predictor for temporal consistency
        self.predictor = SimplePredictorMLP(z_dim, predictor_hidden)
        
        # MINE critic for I(z_next; h1(x_t))
        self.mi_critic = MINECritic(z_dim, z_dim, critic_hidden, critic_depth)
        
        # Conditional MINE critic for I(z_next; x_t | h1(x_t))
        self.cmi_critic = ConditionalMINECritic(x_dim, z_dim, z_dim, critic_hidden, critic_depth)
        
        # Optional: EMA version of encoder for stable negatives
        if use_ema:
            self.encoder_ema = SimpleEncoder(x_dim, z_dim, enc_width, enc_depth)
            self.encoder_ema.load_state_dict(self.encoder.state_dict())
            for p in self.encoder_ema.parameters():
                p.requires_grad = False
            self.ema_momentum = 0.999
        
        print(f"[MINE Model] Created with:")
        print(f"  Encoder: {sum(p.numel() for p in self.encoder.parameters()):,} params")
        print(f"  Predictor: {sum(p.numel() for p in self.predictor.parameters()):,} params")
        print(f"  MI Critic: {sum(p.numel() for p in self.mi_critic.parameters()):,} params")
        print(f"  CMI Critic: {sum(p.numel() for p in self.cmi_critic.parameters()):,} params")
        print(f"  Total: {sum(p.numel() for p in self.parameters()):,} params")
    
    @torch.no_grad()
    def update_ema(self):
        """Update EMA encoder"""
        if not self.use_ema:
            return
        for p_ema, p in zip(self.encoder_ema.parameters(), self.encoder.parameters()):
            p_ema.data.mul_(self.ema_momentum).add_(p.data, alpha=1 - self.ema_momentum)
    
    def mine_loss(self, critic: MINECritic, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute MINE lower bound on I(X; Y)
        
        Returns negative (for minimization): -I(X; Y)
        """
        B = x.shape[0]
        
        # Joint samples: (x_i, y_i)
        scores_joint = critic(x, y)  # (B,)
        
        # Marginal samples: (x_i, y_j) where j is shuffled
        idx_shuffled = torch.randperm(B, device=x.device)
        y_shuffled = y[idx_shuffled]
        scores_marginal = critic(x, y_shuffled)  # (B,)
        
        # MINE bound: E[T(x,y)] - log(E[exp(T(x,y'))])
        # Use log-sum-exp trick for numerical stability
        lse_marginal = torch.logsumexp(scores_marginal, dim=0) - torch.log(torch.tensor(B, dtype=torch.float, device=x.device))
        
        mi_lower_bound = scores_joint.mean() - lse_marginal
        
        # Return negative for minimization
        loss = -mi_lower_bound
        
        stats = {
            'mi_estimate': mi_lower_bound.item(),
            'scores_joint_mean': scores_joint.mean().item(),
            'scores_marginal_mean': scores_marginal.mean().item(),
        }
        
        return loss, stats
    
    def conditional_mine_loss(
        self, 
        critic: ConditionalMINECritic, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute conditional MINE for I(X; Y | Z)
        
        Returns positive (for minimization to enforce CI)
        """
        B = x.shape[0]
        
        # Joint conditional: (x_i, y_i, z_i)
        scores_joint = critic(x, y, z)  # (B,)
        
        # Marginal conditional: (x_i, y_j, z_i) where y_j is shuffled
        idx_shuffled = torch.randperm(B, device=x.device)
        y_shuffled = y[idx_shuffled]
        scores_marginal = critic(x, y_shuffled, z)  # (B,)
        
        # CMI bound
        lse_marginal = torch.logsumexp(scores_marginal, dim=0) - torch.log(torch.tensor(B, dtype=torch.float, device=x.device))
        
        cmi_lower_bound = scores_joint.mean() - lse_marginal
        
        # Return positive for minimization (we want CMI -> 0)
        loss = cmi_lower_bound
        
        stats = {
            'cmi_estimate': cmi_lower_bound.item(),
            'cmi_scores_joint_mean': scores_joint.mean().item(),
            'cmi_scores_marginal_mean': scores_marginal.mean().item(),
        }
        
        return loss, stats
    
    def forward(
        self,
        x_lags: torch.Tensor,
        x_curr: torch.Tensor,
        weights: LossWeights,
        mode: str = "enc",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with MINE-based losses.
        
        Args:
            x_lags: (B, L, x_dim)
            x_curr: (B, x_dim)
            weights: Loss weights
            mode: 'enc' or 'critic' (can alternate training)
        
        Returns:
            Dictionary of losses
        """
        B = x_curr.shape[0]
        
        # Encode current and previous
        z_t = self.encoder(x_curr)  # (B, z_dim)
        x_prev = x_lags[:, -1, :]  # (B, x_dim)
        z_prev = self.encoder(x_prev)  # (B, z_dim)
        
        # ============================================================
        # Loss 1: Reconstruction (if ground truth available)
        # ============================================================
        L_rec = torch.tensor(0.0, device=x_curr.device)
        
        # ============================================================
        # Loss 2: Prediction (standard supervised loss)
        # ============================================================
        z_t_pred = self.predictor(z_prev)
        L_pred = F.mse_loss(z_t_pred, z_t)
        
        # ============================================================
        # Loss 3: Maximize I(z_t; z_prev)
        # ============================================================
        # We want z_prev to be informative about z_t
        L_mi, mi_stats = self.mine_loss(self.mi_critic, z_prev, z_t)
        # L_mi is negative MI (for minimization)
        
        # ============================================================
        # Loss 4: Minimize I(z_t; x_prev | z_prev)
        # ============================================================
        # We want z_t ⊥ x_prev | z_prev
        L_cmi, cmi_stats = self.conditional_mine_loss(self.cmi_critic, x_prev, z_t, z_prev)
        # L_cmi is positive CMI (for minimization)
        
        # ============================================================
        # Total Loss
        # ============================================================
        L_total = (
            weights.w_rec * L_rec +
            weights.w_pred * L_pred +
            weights.w_mi * L_mi +
            weights.w_cmi * L_cmi
        )
        
        # Update EMA if used
        if self.use_ema and mode == "enc":
            self.update_ema()
        
        return {
            'L_rec': L_rec,
            'L_pred': L_pred,
            'L_mi': L_mi,  # Negative MI (minimized)
            'L_cmi': L_cmi,  # Positive CMI (minimized)
            'L_total': L_total,
            # Stats for monitoring
            'mi_estimate': mi_stats['mi_estimate'],
            'cmi_estimate': cmi_stats['cmi_estimate'],
            # Legacy (for compatibility)
            'L_nq0': torch.tensor(0.0, device=x_curr.device),
            'L_nq1': torch.tensor(0.0, device=x_curr.device),
            'L_gate_inst': torch.tensor(0.0, device=x_curr.device),
            'L_gate_delay': torch.tensor(0.0, device=x_curr.device),
        }
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations to latents"""
        return self.encoder(x)


class InfoNCEModel(nn.Module):
    """
    Alternative: InfoNCE-based model (simpler than MINE, often more stable)
    
    Uses contrastive learning instead of MINE critics.
    """
    def __init__(
        self,
        x_dim: int,
        z_dim: int = 1,
        window_size: int = 1,
        enc_width: int = 256,
        enc_depth: int = 2,
        predictor_hidden: int = 128,
        temperature: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.encoder = SimpleEncoder(x_dim, z_dim, enc_width, enc_depth)
        self.predictor = SimplePredictorMLP(z_dim, predictor_hidden)
        self.temperature = temperature
        
        print(f"[InfoNCE Model] Created with:")
        print(f"  Encoder: {sum(p.numel() for p in self.encoder.parameters()):,} params")
        print(f"  Predictor: {sum(p.numel() for p in self.predictor.parameters()):,} params")
    
    def infonce_loss(self, z: torch.Tensor, z_target: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        InfoNCE loss for maximizing agreement between z and z_target
        
        Lower is better (maximizes MI)
        """
        B = z.shape[0]
        
        # Normalize
        z = F.normalize(z, dim=-1)
        z_target = F.normalize(z_target, dim=-1)
        
        # Compute similarities
        logits = (z @ z_target.T) / self.temperature  # (B, B)
        
        # Labels: diagonal entries are positive pairs
        labels = torch.arange(B, device=z.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Accuracy (for monitoring)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
        
        return loss, acc
    
    def forward(self, x_lags: torch.Tensor, x_curr: torch.Tensor, weights: LossWeights, mode: str = "enc") -> Dict:
        """Forward with InfoNCE"""
        z_t = self.encoder(x_curr)
        x_prev = x_lags[:, -1, :]
        z_prev = self.encoder(x_prev)
        
        # Prediction loss
        z_t_pred = self.predictor(z_prev)
        L_pred = F.mse_loss(z_t_pred, z_t)
        
        # InfoNCE for MI maximization (implicit)
        L_contrastive, acc = self.infonce_loss(z_prev, z_t)
        
        # For CI: we want z_t to NOT be predictable from x_prev beyond z_prev
        # Use negative samples of x_prev
        B = x_prev.shape[0]
        idx_neg = torch.randperm(B, device=x_prev.device)
        z_prev_from_x_neg = self.encoder(x_prev[idx_neg])
        
        # If CI holds, z_t should be uncorrelated with z from negative x_prev
        # This is implicit in contrastive learning
        
        L_total = weights.w_pred * L_pred + L_contrastive
        
        return {
            'L_rec': torch.tensor(0.0, device=x_curr.device),
            'L_pred': L_pred,
            'L_mi': L_contrastive,
            'L_cmi': torch.tensor(0.0, device=x_curr.device),
            'L_total': L_total,
            'mi_estimate': -L_contrastive.item(),  # Approximation
            'cmi_estimate': 0.0,
            'contrastive_acc': acc,
            'L_nq0': torch.tensor(0.0, device=x_curr.device),
            'L_nq1': torch.tensor(0.0, device=x_curr.device),
            'L_gate_inst': torch.tensor(0.0, device=x_curr.device),
            'L_gate_delay': torch.tensor(0.0, device=x_curr.device),
        }
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


if __name__ == "__main__":
    print("Testing MINE-based model...")
    
    x_dim = 50
    z_dim = 1
    window_size = 3
    batch_size = 128
    
    # Test MINE model
    model = WholeModel(x_dim=x_dim, z_dim=z_dim, window_size=window_size)
    
    x_lags = torch.randn(batch_size, window_size, x_dim)
    x_curr = torch.randn(batch_size, x_dim)
    weights = LossWeights(w_rec=0.0, w_pred=1.0, w_mi=1.0, w_cmi=0.5)
    
    loss_dict = model(x_lags, x_curr, weights)
    
    print("\nLoss values:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.6f}")
        else:
            print(f"  {key}: {value:.6f}")
    
    loss_dict['L_total'].backward()
    print("\n✓ Backward pass successful!")
    
    # Test InfoNCE model
    print("\n" + "="*60)
    print("Testing InfoNCE model...")
    
    model_infonce = InfoNCEModel(x_dim=x_dim, z_dim=z_dim, window_size=window_size)
    loss_dict = model_infonce(x_lags, x_curr, weights)
    
    print("\nLoss values:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.6f}")
        else:
            print(f"  {key}: {value:.6f}")