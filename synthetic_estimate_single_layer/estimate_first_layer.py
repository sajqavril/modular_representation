import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import copy
from pathlib import Path

# load data

raw_data = np.load('/home/jiaqisun/models/proj_cltd/continue_learning_for_temporal_data/synthetic_estimate_single_layer/data/data_generated/multi_lg_2layer_sigma1.npz')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

z1t0 = torch.tensor(raw_data['zt'][:,0,0], device=device, dtype=torch.float).reshape(-1, 1)
z1t1 = torch.tensor(raw_data['zt'][:,1,0], device=device, dtype=torch.float).reshape(-1, 1)
z2t0 = torch.tensor(raw_data['zt'][:,0,1], device=device, dtype=torch.float).reshape(-1, 1)
z2t1 = torch.tensor(raw_data['zt'][:,1,1], device=device, dtype=torch.float).reshape(-1, 1)
xt0 = torch.tensor(raw_data['xt'][:,0,:], device=device, dtype=torch.float)
xt1 = torch.tensor(raw_data['xt'][:,1,:], device=device, dtype=torch.float)

# define models

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 256, depth: int = 2, act: str = 'gelu'):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, width)
        self.out = nn.Linear(width, out_dim)

        if act == 'gelu':
            self.act = nn.GELU
        elif act == 'tanh':
            self.act = nn.Tanh
        elif act == 'relu':
            self.act = nn.ReLU

        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, width), self.act()) for _ in range(depth)])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = F.dropout(h, 0.2)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)
    
# neural approximator for the conditional densities, theoretical upper bound of the gt CMI for fixed f
class GaussianCondDensityEstimator(nn.Module): 
    def __init__(self, in_dim, out_dim, hid_dim, hid_layers, LOGVAR_MIN=-5.0, LOGVAR_MAX=2.0, learn_logvar=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.hid_layers = hid_layers

        self.mu = MLP(in_dim=self.in_dim,
                      out_dim=self.out_dim,
                      width=self.hid_dim,
                      depth=self.hid_layers)
        self.logvar = MLP(in_dim=self.in_dim,
                      out_dim=self.out_dim,
                      width=self.hid_dim,
                      depth=self.hid_layers)
        self.LOGVAR_MIN = LOGVAR_MIN
        self.LOGVAR_MAX = LOGVAR_MAX
        
        if learn_logvar == False:
            self.LOGVAR_MIN = 0.
            self.LOGVAR_MAX = 0.
        
    def forward(self, x, cond): # p(x|cond)
        '''
        x: N \times d_x
        cond: N \times d_cond
        '''
        mu_x_cond = self.mu(cond)
        logvar = self.logvar(cond)
        logvar = torch.clamp(logvar, min=self.LOGVAR_MIN, max=self.LOGVAR_MAX)
        mu_var_cond = torch.exp(logvar)
        x_mu = x - mu_x_cond.reshape(-1, self.out_dim)  
        
        logp = -0.5 * self.out_dim * torch.tensor(2*torch.pi, device=cond.device, dtype=cond.dtype).log() \
            -0.5 * torch.log(mu_var_cond).sum(dim=-1)\
                -0.5 * (x_mu**2 * (1.0 / mu_var_cond)).sum(dim=-1)    
        return logp

# Encoder f into to the training process
class EncoderF1(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, hid_layers, linear=False, act='gelu'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.hid_layers = hid_layers
        self.linear = linear

        if not self.linear:
            self.encoder_f1 = MLP(in_dim=self.in_dim,
                              out_dim=self.out_dim,
                              width=self.hid_dim,
                              depth=self.hid_layers,
                              act=act)
        
        else:
            self.encoder_f1 = nn.Linear(in_features=self.in_dim,
                                        out_features=self.out_dim)
    def forward(self, x):
        z = self.encoder_f1(x)
        return z


def state_dict_to_numpy(state_dict):
    return {key: value.detach().cpu().numpy() for key, value in state_dict.items()}


def save_model_bundle_npz(save_path, models, metadata):
    save_dict = {}

    for model_name, model in models.items():
        for key, value in state_dict_to_numpy(model.state_dict()).items():
            save_dict[f'{model_name}.{key}'] = value

    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            save_dict[key] = value.detach().cpu().numpy()
        else:
            save_dict[key] = np.asarray(value)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, **save_dict)


# Train the first-layer encoder together with the CMI estimator.
def train():
    lr_q = 1e-3
    lr_f = 3e-4
    lr_pred = 3e-4

    lambda_jepa = 0.1   # 
    lambda_recon = 0.1
    lambda_cmi = 1e-2

    n_epochs = 1000
    n_q_epochs = 20
    n_f_epochs = 3
    hid_dim = 64
    hid_layer = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_samples, _, n = raw_data['zt'].shape
    d = raw_data['xt'].shape[-1]
    linear = False
    save_path = Path(__file__).with_name('saved_models') / 'estimate_first_layer_model.npz'

    # JEPA / teacher params
    ema_m = 0.95          # teacher momentum
    use_jepa_warmup = True
    warmup_epochs = 50    # JEPA-only warmup

    f1s = [] # save the history of 

    q_z1t1_z1t0 = GaussianCondDensityEstimator(
        in_dim=1,
        out_dim=1,
        hid_dim=hid_dim,
        hid_layers=hid_layer,
        learn_logvar=False
    ).to(device)

    q_z1t1_xt0 = GaussianCondDensityEstimator(
        in_dim=d,
        out_dim=1,
        hid_dim=hid_dim,
        hid_layers=hid_layer,
        learn_logvar=False
    ).to(device)

    f1 = EncoderF1(
        in_dim=d,
        out_dim=1,
        hid_dim=hid_dim,
        hid_layers=hid_layer,
        linear=linear,
        act='gelu'
    ).to(device)

    # teacher encoder: same architecture, no grad
    f1_teacher = copy.deepcopy(f1).to(device)
    for p in f1_teacher.parameters():
        p.requires_grad = False

    # small predictor q on latent space
    # yes, this is "in latent space"; since out_dim=1, keep it tiny
    q_pred = nn.Sequential(
        nn.Linear(1, 2),
        nn.GELU(),
        nn.Linear(2, 1)
    ).to(device)

    # keep g1 if you want, but it is no longer needed for JEPA loss
    g1 = EncoderF1(
        in_dim=1,
        out_dim=d,
        hid_dim=hid_dim,
        hid_layers=hid_layer,
        linear=linear,
        act='gelu'
    ).to(device)

    optim_q = torch.optim.Adam(
        itertools.chain(
            q_z1t1_xt0.parameters(),
            q_z1t1_z1t0.parameters()
        ),
        lr=lr_q
    )

    optim_f = torch.optim.Adam(
        itertools.chain(
            f1.parameters(),
            g1.parameters()
        ),
        lr=lr_f
    )

    optim_pred = torch.optim.Adam(
        q_pred.parameters(),
        lr=lr_pred
    )

    # Normalize x once and reuse the same stats at t and t+1.
    x_mean = xt0.mean(dim=0, keepdim=True)
    x_std = xt0.std(dim=0, keepdim=True) + 1e-8
    xt0 = (xt0 - x_mean) / x_std
    xt1 = (xt1 - x_mean) / x_std

    for epoch in range(n_epochs):
        if (epoch + 1) % 20 == 0:
            print(f'*** Epoch: {epoch+1}/{n_epochs} ***')

        # Update the conditional density estimators using the current encoder.
        q_z1t1_xt0.train()
        q_z1t1_z1t0.train()

        est_z1t0 = f1(xt0)
        est_z1t1 = f1(xt1)

        for q_epoch in range(n_q_epochs):
            optim_q.zero_grad()

            logp_q_z1t1_xt0 = q_z1t1_xt0(x=est_z1t1.detach(), cond=xt0)
            logp_q_z1t1_z1t0 = q_z1t1_z1t0(x=est_z1t1.detach(), cond=est_z1t0.detach())
            estimator_loss = -logp_q_z1t1_xt0.mean() - logp_q_z1t1_z1t0.mean()
            cmi = (logp_q_z1t1_xt0 - logp_q_z1t1_z1t0).mean()

            if ((q_epoch + 1) % (n_q_epochs//2) == 0) and ((epoch + 1) % 20 == 0):
                print(
                    f'q Epoch [{q_epoch+1}/{n_q_epochs}], cmi:{cmi.detach().item():.4f}, '
                    f'q_up:{logp_q_z1t1_xt0.mean().detach().item():.4f}, '
                    f'q_down:{logp_q_z1t1_z1t0.mean().detach().item():.4f}'
                )

            estimator_loss.backward()
            optim_q.step()

        # Then update the encoder, predictor, and decoder-side regularizers.
        for f_epoch in range(n_f_epochs):
            optim_f.zero_grad()
            optim_pred.zero_grad()

            est_z1t0 = f1(xt0)
            est_z1t1 = f1(xt1)

            # JEPA target from teacher; sg = detach / no-grad
            with torch.no_grad():
                tgt_z1t1 = f1_teacher(xt1)

            pred_z1t1 = q_pred(est_z1t0)

            recon_loss = F.mse_loss(g1(est_z1t0), xt0)

            mi_reg = F.mse_loss(pred_z1t1, tgt_z1t1)

            logp_q_z1t1_xt0 = q_z1t1_xt0(x=est_z1t1, cond=xt0)
            logp_q_z1t1_z1t0 = q_z1t1_z1t0(x=est_z1t1, cond=est_z1t0)

            cmi_loss = (logp_q_z1t1_xt0 - logp_q_z1t1_z1t0).mean()

            var_reg = F.relu(0.5 - est_z1t0.std()).mean()

            # warmup: JEPA first, then combine
            if use_jepa_warmup and epoch < warmup_epochs:
                loss_f = lambda_jepa * mi_reg + lambda_recon * recon_loss # make it better to anchor the x subspace
                if epoch == warmup_epochs - 1:
                    with torch.no_grad():
                        pred0 = F.mse_loss(q_pred(f1(xt0)), f1_teacher(xt1)).item()
                        pred_tol = 1.0 * pred0
    
            else:
                mi_reg = F.relu(mi_reg - pred_tol)
                loss_f = cmi_loss * lambda_cmi + lambda_jepa * mi_reg + 0.1 * var_reg

            if ((f_epoch + 1) % 1 == 0) and ((epoch + 1) % 20 == 0):
                print(
                    f'f Epoch [{f_epoch+1}/{n_f_epochs}], '
                    f'cmi:{cmi_loss.detach().item():.4f}, '
                    f'jepa:{mi_reg.detach().item():.4f}, '
                    f'q_up:{logp_q_z1t1_xt0.mean().detach().item():.4f}, '
                    f'q_down:{logp_q_z1t1_z1t0.mean().detach().item():.4f}'
                )
                f1s.append(copy.deepcopy(f1))
                with torch.no_grad():
                    print(
                        'std(est_z1t0)=', est_z1t0.std().item(),
                        'corr_z1=',
                        torch.corrcoef(torch.cat([est_z1t0.squeeze(), z1t0.squeeze()], dim=0).view(2, -1))[0,1].item(),
                        'corr_z2=',
                        torch.corrcoef(torch.cat([est_z1t0.squeeze(), z2t0.squeeze()], dim=0).view(2, -1))[0,1].item()
                    )

            loss_f.backward()
            optim_f.step()
            optim_pred.step()

            # EMA teacher update
            with torch.no_grad():
                for p_t, p_s in zip(f1_teacher.parameters(), f1.parameters()):
                    p_t.data.mul_(ema_m).add_(p_s.data, alpha=1.0 - ema_m)

    save_model_bundle_npz(
        save_path=save_path,
        models={
            'f1': f1,
            'f1_teacher': f1_teacher,
            'q_pred': q_pred,
            'g1': g1,
            'q_z1t1_xt0': q_z1t1_xt0,
            'q_z1t1_z1t0': q_z1t1_z1t0,
        },
        metadata={
            'x_mean': x_mean,
            'x_std': x_std,
            'hid_dim': hid_dim,
            'hid_layer': hid_layer,
            'n_epochs': n_epochs,
            'n_q_epochs': n_q_epochs,
            'n_f_epochs': n_f_epochs,
            'lambda_jepa': lambda_jepa,
            'lambda_recon': lambda_recon,
            'lambda_cmi': lambda_cmi,
            'ema_m': ema_m,
        },
    )
    print(f'Saved model bundle to: {save_path}')

    return {
        'f1': f1,
        'f1_teacher': f1_teacher,
        'q_pred': q_pred,
        'g1': g1,
        'q_z1t1_xt0': q_z1t1_xt0,
        'q_z1t1_z1t0': q_z1t1_z1t0,
        'save_path': str(save_path),
    }


if __name__ == '__main__':
    train()
