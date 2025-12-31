# Estimate zt(1) by aligning mutual info when assuming additive noise
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models import SeqEstARModel_PreZT, SeqEstARModel_CurrentPreZT
from utils import entropy_kde_torch, entropy_gaussian_torch, generate_lambda_choices

def train(xt, l_f_list, l_m_list, l_mi_list, nlags, hid_dim, runs, device, lr=0.001, wd=5e-5, epochs=10_000, option=1):
    print(f"[TRAIN] Start training for nlags={nlags}, hid_dim={hid_dim}")
    length, xt_dim, batch_size = xt.shape
    pre_zt_list = []
    for layer in range(xt_dim):
        print(f"[TRAIN] Estimating latent layer {layer}")
        best_pre_zt = 0
        best_loss = 999999
        for run in range(runs):
            print(f"  [TRAIN] Run {run+1}/{runs} for layer {layer}")
            if option == 2:
                layer_model = SeqEstARModel_CurrentPreZT(
                    nlags=nlags, 
                    xt_dim=xt_dim,
                    est_zt_dim=1,
                    hid_dim=hid_dim,
                    n_pre_zt=len(pre_zt_list)
                ).to(device)
            else:
                layer_model = SeqEstARModel_PreZT(
                    nlags=nlags, 
                    xt_dim=xt_dim,
                    est_zt_dim=1,
                    hid_dim=hid_dim,
                    n_pre_zt=len(pre_zt_list)
                ).to(device)

            layer_model.train()
            optimizer = torch.optim.Adam(params=layer_model.parameters(), lr=lr, weight_decay=wd)
            recon_loss_fn = torch.nn.MSELoss(reduction='mean')

            for epoch in range(epochs):
                zt_p1, f_zt_p1, h_m_xt_p1, h_inv_zt_p1 = layer_model(xt, pre_zt_list)
                loss_recon_m = recon_loss_fn(zt_p1, h_m_xt_p1)
                loss_recon_f = recon_loss_fn(zt_p1, f_zt_p1)
                loss_mi = recon_loss_fn(xt[nlags:, :, :], h_inv_zt_p1)
                entropy_f = entropy_gaussian_torch(zt_p1 - f_zt_p1)
                entropy_m = entropy_gaussian_torch(zt_p1 - h_m_xt_p1)
                loss_ci = entropy_f - entropy_m

                loss = loss_ci + l_mi_list[layer] * loss_mi + l_f_list[layer] * loss_recon_f + l_m_list[layer] * loss_recon_m

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"    [TRAIN] Run {run+1} finished, loss={loss.item():.6f}")
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_pre_zt = layer_model.compute_h_xt(xt).detach()

        with torch.no_grad():
            pre_zt_list.append(best_pre_zt)        
        print(f"[TRAIN] Finished layer {layer}, best loss={best_loss:.6f}")

    print("[TRAIN] Training completed for all layers")
    return pre_zt_list


def evaluate(est_zt_list, zt):
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
    parser.add_argument("--data-dir", type=str, default="datasets", help='the path of the dir of the synthetic data; all datasets will be processed')
    parser.add_argument("--runs", type=int, default=5, help='the number of runs to avoid local solutions on each synthetic dataset')
    parser.add_argument("--res-dir", type=str, default="est_figures", help='the path to save the output figures')    
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

        # Generate lambda choices once
        n_layers = nlatents  # number of latents = number of layers for lambda
        max_choices = 7
        print("[MAIN] Generating lambda choices")
        l_f_choices = generate_lambda_choices(n_layers, max_choices)
        l_m_choices = generate_lambda_choices(n_layers, max_choices)
        l_mi_choices = generate_lambda_choices(n_layers, max_choices)

        data = np.load(os.path.join(args.data_dir, filename))
        xt = torch.tensor(data['xt'], dtype=torch.float32, device=device)
        zt = torch.tensor(data['zt'], dtype=torch.float32, device=device)
        print(f"[MAIN] Loaded data shapes xt={xt.shape}, zt={zt.shape}")

        nlags_list = list(range(max(gt_lags-2, 1), length-1))
        hid_dim_list = [1, 8, 16]
        f, axs = plt.subplots(nrows=len(nlags_list), ncols=len(hid_dim_list), figsize=(len(hid_dim_list)*5, len(nlags_list)*4))
        axs = np.atleast_2d(axs)

        for row, nlags in enumerate(nlags_list):
            for col, hid_dim in enumerate(hid_dim_list):
                ax = axs[row, col]
                print(f"[MAIN] Running nlags={nlags}, hid_dim={hid_dim}")
                for lambda_choice in range(len(l_f_choices)):
                    print(f"  [MAIN] Lambda choice {lambda_choice}")
                    l_f_list = l_f_choices[lambda_choice]
                    l_m_list = l_m_choices[lambda_choice]
                    l_mi_list = l_mi_choices[lambda_choice]

                    # Train the model
                    est_zt_list = train(
                        xt=xt,
                        l_f_list=l_f_list,
                        l_m_list=l_m_list,
                        l_mi_list=l_mi_list,
                        nlags=nlags,
                        hid_dim=hid_dim,
                        runs=args.runs,
                        device=device,
                        option=args.option
                    )
                    l2_dist_list = evaluate(est_zt_list=est_zt_list, zt=zt)
                    ax.plot(l2_dist_list, alpha=0.5, marker="*", label=f"lambda-{lambda_choice}")
                ax.set_xticks(list(range(nlatents)))
                ax.set_xlabel("Latents Index")
                ax.set_ylabel("L2 dist between est and gt")
                ax.legend()

        f.suptitle(filename)
        os.makedirs(args.res_dir, exist_ok=True)
        out_path = os.path.join(args.res_dir, f"{filename.replace('.npz', '')}.png")
        f.savefig(out_path)
        print(f"[MAIN] Saved figure to {out_path}")


if __name__ == "__main__":
    main()
