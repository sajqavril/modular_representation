# Estimate zt(1) by aligning mutual info when assuming additive noise
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models import ComplexARModel

def train(model, xt, alpha, beta, lr=0.001, wd=5e-5, epochs=10_000):
    model.train()
    optimizer =  torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    for epoch in range(epochs):
        xt_recon, zt_recon, zt_recon_Hzt, zt_recon_Hxt = model(xt)
        length_for_loss = xt_recon.shape[0]
        ar_loss_zt = loss_fn(zt_recon_Hzt, zt_recon[-length_for_loss:,:])
        ar_loss_xt = loss_fn(xt_recon, xt[-length_for_loss,:,:])
        mi_loss = (loss_fn(zt_recon_Hxt, zt_recon[-length_for_loss:,:]) - ar_loss_zt) ** 2 / 2
        loss = mi_loss + alpha * ar_loss_zt + beta * ar_loss_xt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def evaluate(model, xt, zt):
    model.eval()
    xt_recon, zt_recon, zt_recon_Hzt, zt_recon_Hxt = model(xt)
    length_for_loss = zt_recon.shape[0]
    gt_zt = zt[-length_for_loss:,:]

    # normalization
    zt_normed = zt - zt.mean(dim=-1, keepdim=True)
    gt_zt_normed = gt_zt - gt_zt.mean(dim=-1, keepdim=True)
    similarities = ((zt - gt_zt.unsqeeze(dim=1)) ** 2).mean(dim=(0,2))

    return similarities.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset", help='the path of the dir of the synthetic data; all datasets will be processed')
    parser.add_argument("--runs", type=int, default=5, help='the number of runs to avoid local solutions on each synthetic dataset')
    parser.add_argument("--res-dir", type=str, default="est_figures", help='the path to save the output figures')
    parser.add_argument("--alpha", type=float, default=0.5, help="weight for ar_zt term in the loss")
    parser.add_argument("--beta", type=float, default=0.5, help="weight for ar_xt term in the loss")
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)

    assert os.path.exists(args.data_dir)
    filenames = os.listdir(args.data_dir)
    for raw_filename in filenames:
        filename = raw_filename.split(".")[0]
        gt_lags = int(filename.split("nlags=")[1].split("_")[0])
        length = int(filename.split("length=")[1].split("_")[0])
        nlatents = int(filename.split("nlatents=")[1].split("_")[0])

        data = np.load(os.path.join(args.data_dir, filename))
        xt = data['xt']
        zt = data['zt']

        nlags_list = list(range(min(gt_lags-2, 1), length-1))
        hid_dim_list = [1, 8, 16]
        f, axs = plt.subplots(nrows=len(nlags_list), ncols=len(hid_dim_list), figsize=(len(hid_dim_list)*5, len(nlags_list)*4))
        for row, nlags in enumerate(nlags_list):
            for col, hid_dim in enumerate(hid_dim_list):
                ax = axs[row, col]
                best_loss = 9999
                best_model = 0
                for run in range(args.runs):
                    model = ComplexARModel(nlags=nlags, in_dim_xt=nlatents, in_dim_zt=1, hid_dim=hid_dim)
                    loss = train(model, xt, alpha=args.alpha, beta=args.beta)
                    if loss < best_loss:
                        best_model = model
                sim_list = evaluate(best_model, xt, zt)
                ax.plot(sim_list, alpha=0.5, marker="*")
                ax.set_xticks(list(range(nlatents)))
                ax.set_xlabel("Latents Index")
                ax.set_ylabel("Similarity")

        f.suptitle(filename)
        os.makedirs(args.res_dir, exist_ok=True)
        f.savefig(os.path.join(args.res_dir, f"{filename}_alpha={args.alpha}_beta={args.beta}.png"))


if __name__ == "__main__":
    main()