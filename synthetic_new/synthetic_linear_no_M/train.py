# train.py
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from model import LinearModelNoMforCL
from utils import (
    generate_multiview_x_datasets,
    make_dataloader_for_k,
    whiteness_loss,
    seed_everything
)


@dataclass
class TrainCfg:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs_per_layer: int = 2

    # losses
    lambda_recon: float = 1.0
    lambda_offdiag: float = 0.5   # cross-dim decorrelation
    lambda_lag1: float = 0.5      # temporal lag-1 decorrelation
    lambda_eps_energy: float = 0.0  # optional: small eps energy

    # data loader behavior
    shuffle: bool = True  # set False if you really want lag-1 penalty to fire often
    
    # seed
    seed: int = 42 


def train_progressive(model: LinearModelNoMforCL, x_dict, L: int, cfg: TrainCfg):
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for active_layer in range(model.nlayers):
        k = (active_layer + 1) * model.block
        loader = make_dataloader_for_k(
            x_dict=x_dict,
            k=k,
            L=L,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=0,
            drop_last=True,
            seed=cfg.seed
        )

        model.set_active_layer(active_layer)

        for epoch in range(cfg.epochs_per_layer):
            for x_lags, x_curr, t_idx in loader:
                x_lags = x_lags.to(cfg.device)         # (B,L,n)
                x_curr = x_curr.to(cfg.device)         # (B,1,n)
                t_idx = t_idx.to(cfg.device)           # (B,)

                out = model.mask_model(x_lags, x_curr, active_layer)

                # recon on x_curr from z_pred
                loss_recon = F.mse_loss(out["x_pred"], x_curr)

                # whiteness / independence-ish loss on eps_hat
                loss_white, white_stats = whiteness_loss(
                    eps_hat=out["eps_hat"],
                    t_idx=t_idx if not cfg.shuffle else None,  # lag-1 needs adjacency; shuffle breaks it
                    lambda_offdiag=cfg.lambda_offdiag,
                    lambda_lag1=cfg.lambda_lag1,
                )

                # optional: keep eps_hat energy bounded (can help early training)
                loss_eps_energy = out["eps_hat"].pow(2).mean()

                loss = (
                    cfg.lambda_recon * loss_recon
                    + loss_white
                    + cfg.lambda_eps_energy * loss_eps_energy
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                # enforce B structure strictly (especially if weight_decay>0)
                model.project_Bs()

            print(
                f"[layer={active_layer} k={k}] epoch {epoch+1}/{cfg.epochs_per_layer} "
                f"recon={loss_recon.item():.4e} "
                f"white(offdiag={white_stats['loss_offdiag']:.3e}, lag1={white_stats['loss_lag1']:.3e}, pairs={white_stats['lag1_pairs']}) "
                f"epsE={loss_eps_energy.item():.3e}"
            )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlayer", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs_per_layer", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--shuffle", action="store_true", help="shuffle training windows (lag1 whiteness will be disabled)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="data/uniform_3_same.npz")
    args = parser.parse_args()

    seed_everything(args.seed)

    data = np.load(args.data_path, allow_pickle=True)
    x = data["xt"]        # shape: (T, n)
    z = data["zt"]        # shape: (T, d)
    B_list = data["B_list"]  # shape: (L, d, d)
    A = data["A"]         # shape: (d, d)
    T, n = x.shape
    T, d = z.shape
    L = len(B_list)

    x_dict = generate_multiview_x_datasets(z=z, A=A, save_dir="./data", save_npz=False)

    model = LinearModelNoMforCL(n=n, d=d, nlayer=args.nlayer, L=L)

    cfg = TrainCfg(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs_per_layer=args.epochs_per_layer,
        shuffle=args.shuffle,
        # You can tune these:
        lambda_offdiag=0.5,
        lambda_lag1=0.5,
        lambda_eps_energy=0.0,
        seed=args.seed
    )

    train_progressive(model, x_dict, L=L, cfg=cfg)

    print("Output correlation matrix between z and \hat z: ")
    print(model.g_inv.weight.detach().numpy() @ np.linalg.pin(A))

    print(f'GT Bs: {B_list}')
    for i, B in enumerate(model.Bs):
        print(f"B[{i}] weight:\n{B.weight.data}")


if __name__ == "__main__":
    main()
