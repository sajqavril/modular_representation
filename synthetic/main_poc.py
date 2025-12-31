# Proof of Concept training script using adaptive hyperparameters
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models import SeqEstARModel_CurrentPreZT, SeqEstARModel_PreZT
from utils import entropy_gaussian_torch


def train(xt, zt, nlags, hid_dim, runs, device, lr=0.001, wd=5e-5, epochs=10_000, option=1):
    """
    Proof of Concept Training:
    - Uses adaptive lambda weights (l_f, l_m, l_mi) instead of manual sweeping.
    - Minimizes (loss_ci + loss_mi) and tracks minimum values per latent layer.
    """
    print(f"[TRAIN] Start POC training for nlags={nlags}, hid_dim={hid_dim}")
    length, xt_dim, batch_size = xt.shape
    pre_zt_list = []
    min_loss_curve = []
    ci_curves = []
    mi_curves = []

    # Initialize lambdas uniformly
    l_f, l_m, l_mi = 1.0, 1.0, 1.0

    for layer in range(xt_dim):
        print(f"[TRAIN] Estimating latent layer {layer}")
        best_pre_zt = 0
        best_loss = float("inf")
        best_min_ci_mi = float("inf")
        best_ci = float("inf")
        best_mi = float("inf")
        zt_layer = zt[:,layer,:].unsqueeze(dim=1)

        for run in range(runs):
            print(f"  [TRAIN] Run {run+1}/{runs} for layer {layer}")
            if option == 2:
                layer_model = SeqEstARModel_CurrentPreZT(
                    nlags=nlags, 
                    xt_dim=xt_dim,
                    est_zt_dim=1,
                    hid_dim=hid_dim,
                    n_pre_zt=len(pre_zt_list),
                    poc_flag=True
                ).to(device)
            else:
                layer_model = SeqEstARModel_PreZT(
                    nlags=nlags, 
                    xt_dim=xt_dim,
                    est_zt_dim=1,
                    hid_dim=hid_dim,
                    n_pre_zt=len(pre_zt_list),
                    poc_flag=True
                ).to(device)

            layer_model.train()
            optimizer = torch.optim.Adam(params=layer_model.parameters(), lr=lr, weight_decay=wd)
            recon_loss_fn = torch.nn.MSELoss(reduction='mean')

            for epoch in range(epochs):
                # === Forward pass ===
                zt_p1, f_zt_p1, h_m_xt_p1, h_inv_zt_p1 = layer_model(xt, pre_zt_list=pre_zt_list, zt=zt_layer)

                # === Loss components ===
                loss_recon_m = recon_loss_fn(zt_p1, h_m_xt_p1)
                loss_recon_f = recon_loss_fn(zt_p1, f_zt_p1)
                loss_mi = recon_loss_fn(xt[nlags:, :, :], h_inv_zt_p1)
                entropy_f = entropy_gaussian_torch(zt_p1 - f_zt_p1)
                entropy_m = entropy_gaussian_torch(zt_p1 - h_m_xt_p1)
                loss_ci = entropy_f - entropy_m

                # === Total loss ===
                loss = l_f * loss_recon_f + l_m * loss_recon_m + l_mi * loss_mi + loss_ci

                # === Track min(loss_ci + loss_mi) ===
                ci_mi_val = (loss_ci + loss_mi).item()
                if ci_mi_val < best_min_ci_mi:
                    best_min_ci_mi = ci_mi_val

                # === Backprop ===
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # === Adaptive lambda updates every 200 epochs ===
                if epoch % 200 == 0 and epoch > 0:
                    raw_lf = 1.0 / (loss_recon_f.item() + 1e-6)
                    raw_lm = 1.0 / (loss_recon_m.item() + 1e-6)
                    raw_lmi = 1.0 / (loss_mi.item() + 1e-6)

                    total = raw_lf + raw_lm + raw_lmi
                    l_f = 3.0 * raw_lf / total
                    l_m = 3.0 * raw_lm / total
                    l_mi = 3.0 * raw_lmi / total
                    
                    print(f"        [TRAIN] Epoch {epoch+1} | total loss={loss.item():.4f} | loss_ci={loss_ci.item():.4f} | loss_mi={loss_mi.item():.4f} | loss_recon_f={loss_recon_f.item():.4f} | loss_recon_m={loss_recon_m.item():.4f}")


            print(f"    [TRAIN] Run {run+1} finished, final loss={loss.item():.6f}, min(ci+mi)={best_min_ci_mi:.6f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_pre_zt = layer_model.compute_h_xt(xt).detach()
                best_ci = loss_ci.item()
                best_mi = loss_mi.item()

        min_loss_curve.append(best_min_ci_mi)
        ci_curves.append(best_ci)
        mi_curves.append(best_mi)

        with torch.no_grad():
            pre_zt_list.append(best_pre_zt)
        print(f"[TRAIN] Finished layer {layer}, best loss={best_loss:.6f}, min(ci+mi)={best_min_ci_mi:.6f}")

    print("[TRAIN] POC training completed")
    return pre_zt_list, (min_loss_curve, ci_curves, mi_curves)


def evaluate(est_zt_list, zt):
    """Compute L2 distance between estimated and ground-truth latent variables."""
    print("[EVAL] Start evaluation")
    l2_dist_list = []
    for layer, est_zt in enumerate(est_zt_list):
        gt_zt = zt[:, layer, :]
        l2_dist = (gt_zt - est_zt).norm(p=2).item()
        l2_dist_list.append(l2_dist)
        print(f"  [EVAL] Layer {layer} L2 distance: {l2_dist:.6f}")
    print("[EVAL] Evaluation completed")
    return l2_dist_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="datasets")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--res-dir", type=str, default="poc_figures")
    parser.add_argument("--option", type=int, default=1, choices=[1, 2])

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    print(f"[MAIN] Using device: {device}")

    assert os.path.exists(args.data_dir)
    filenames = os.listdir(args.data_dir)

    for filename in filenames:
        print(f"[MAIN] Processing file: {filename}")
        gt_lags = int(filename.split("nlags=")[1].split("_")[0])
        length = int(filename.split("length=")[1].split("_")[0])
        nlatents = int(filename.split("nlatents=")[1].split("_")[0])

        data = np.load(os.path.join(args.data_dir, filename))
        xt = torch.tensor(data['xt'], dtype=torch.float32, device=device)
        zt = torch.tensor(data['zt'], dtype=torch.float32, device=device)
        print(f"[MAIN] Loaded data shapes xt={xt.shape}, zt={zt.shape}")

        # Define nlags & hid_dim choices
        nlags_list = list(range(max(gt_lags - 2, 1), length - 1))
        hid_dim_list = [1, 8, 16]

        # Prepare plotting grid 
        # 2 figures: ci and mi
        for figure_idx in range(2):
            f, axs = plt.subplots(
                nrows=len(nlags_list),
                ncols=len(hid_dim_list),
                figsize=(len(hid_dim_list) * 5, len(nlags_list) * 4)
            )
            
            axs = np.atleast_2d(axs)

            for row, nlags in enumerate(nlags_list):
                for col, hid_dim in enumerate(hid_dim_list):
                    ax = axs[row, col]
                    print(f"[MAIN] Running nlags={nlags}, hid_dim={hid_dim}")

                    # Train the model with adaptive lambdas
                    est_zt_list, (ci_mi_curves, ci_curves, mi_curves) = train(
                        xt=xt,
                        zt=zt,
                        nlags=nlags,
                        hid_dim=hid_dim,
                        runs=args.runs,
                        device=device,
                        option=args.option
                    )

                    # Evaluate quality of estimated zt
                    evaluate(est_zt_list=est_zt_list, zt=zt)

                    if figure_idx == 0:
                        ax.plot(ci_curves, marker="*", alpha=0.7)
                    else:
                        ax.plot(mi_curves, marker="*", alpha=0.7)
                    ax.set_xticks(list(range(nlatents)))
                    ax.set_xlabel("Latent Index")
                    if figure_idx == 0:
                        ax.set_ylabel("Loss CI")
                    else:
                        ax.set_ylabel("Loss MI")
                    ax.set_title(f"nlags-{nlags}, hid_dim-{hid_dim}")

            if figure_idx == 0:
                f.suptitle(f"POC CI Results - {filename}")
            else:
                f.suptitle(f"POC MI Results - {filename}")
            os.makedirs(args.res_dir, exist_ok=True)
            if figure_idx == 0:
                out_path = os.path.join(args.res_dir, f"{filename.replace('.npz', '')}_poc_ci.png")
            else:
                out_path = os.path.join(args.res_dir, f"{filename.replace('.npz', '')}_poc_mi.png")

            f.savefig(out_path)
            print(f"[MAIN] Saved POC figure to {out_path}")


if __name__ == "__main__":
    main()
