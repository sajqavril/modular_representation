# Proof of concept: whether z1 has the highest self-reconstruction performance
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models import SimpleARModel

def train(model, zit, lr=0.001, wd=5e-5, epochs=10_000):
    # zit is of shape (length, sample_size)
    zit = np.expand_dims(zit, 1)
    optimizer =  torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    for epoch in range(epochs):
        recon_zit = model(zit)
        length_for_loss = recon_zit.shape[0]
        loss = loss_fn(recon_zit, zit[-length_for_loss:])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset", help='the path of the dir of the synthetic data; all datasets will be processed')
    parser.add_argument("--runs", type=int, default=5, help='the number of runs to avoid local solutions on each synthetic dataset')
    parser.add_argument("--res-dir", type=str, default="poc_figures", help='the path to save the output figures')
    
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
        zt = data['zt']

        nlags_list = list(range(min(gt_lags-2, 1), length-1))
        hid_dim_list = [1, 8, 16]
        f, axs = plt.subplots(nrows=len(nlags_list), ncols=len(hid_dim_list), figsize=(len(hid_dim_list)*5, len(nlags_list)*4))
        for row, nlags in enumerate(nlags_list):
            for col, hid_dim in enumerate(hid_dim_list):
                ax = axs[row, col]
                zt_loss_list = []
                for z in range(nlatents):
                    z_loss = 999999
                    for run in range(args.runs):
                        model = SimpleARModel(nlags=nlags, in_dim=1, hid_dim=hid_dim)
                        _best_loss = train(model, zt[:, z, :])
                        z_loss = min(z_loss, _best_loss)
                    zt_loss_list.append(z_loss)

                ax.plot(zt_loss_list, alpha=0.5, marker="o")
                ax.set_xticks(list(range(nlatents)))
                ax.set_xlabel("Latents Index")
                ax.set_ylabel("Loss")

        f.suptitle(filename)
        os.makedirs(args.res_dir, exist_ok=True)
        f.savefig(os.path.join(args.res_dir, f"{filename}.png"))


if __name__ == "__main__":
    main()