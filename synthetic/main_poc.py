# Proof of concept: whether z1 has the highest self-reconstruction performance
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models import SimpleARModel

def train(model, zit, lr=0.001, wd=5e-5, epochs=10_000):
    # zit is of shape (length, sample_size)
    print(f"  [train] Input zit shape (before expand): {zit.shape}")
    zit = np.expand_dims(zit, 1)  # -> (length, 1, sample_size)
    zit = torch.tensor(zit, dtype=torch.float32, device=next(model.parameters()).device)
    print(f"  [train] Input zit shape (torch tensor): {zit.shape}")

    optimizer =  torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    for epoch in range(epochs):
        recon_zit = model(zit).permute(0, 2, 1)
        length_for_loss = recon_zit.shape[0]
        loss = loss_fn(recon_zit, zit[-length_for_loss:])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % (epochs // 5) == 0:  # log 5 times per training
            print(f"    [train] Epoch {epoch+1}/{epochs}, loss={loss.item():.6f}")

    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="datasets", help='the path of the dir of the synthetic data; all datasets will be processed')
    parser.add_argument("--runs", type=int, default=5, help='the number of runs to avoid local solutions on each synthetic dataset')
    parser.add_argument("--res-dir", type=str, default="poc_figures", help='the path to save the output figures')
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    print(f"[main] Using device: {device}")

    assert os.path.exists(args.data_dir), f"Data dir {args.data_dir} does not exist"
    filenames = os.listdir(args.data_dir)
    print(f"[main] Found {len(filenames)} dataset files in {args.data_dir}")

    for filename in filenames:
        print(f"\n[main] Processing file: {filename}")

        # parse metadata from filename
        gt_lags = int(filename.split("nlags=")[1].split("_")[0])
        length = int(filename.split("length=")[1].split("_")[0])
        nlatents = int(filename.split("nlatents=")[1].split("_")[0])
        print(f"  [main] Parsed: gt_lags={gt_lags}, length={length}, nlatents={nlatents}")

        # load dataset
        data = np.load(os.path.join(args.data_dir, filename))
        zt = data['zt']
        print(f"  [main] Loaded zt with shape {zt.shape}")

        nlags_list = list(range(max(gt_lags-2, 1), length-1))
        hid_dim_list = [1, 8, 16]
        print(f"  [main] Running experiments over nlags_list={nlags_list}, hid_dim_list={hid_dim_list}")

        f, axs = plt.subplots(nrows=len(nlags_list), ncols=len(hid_dim_list), figsize=(len(hid_dim_list)*5, len(nlags_list)*4))
        for row, nlags in enumerate(nlags_list):
            for col, hid_dim in enumerate(hid_dim_list):
                ax = axs[row, col] if len(nlags_list) > 1 else axs[col]
                print(f"    [loop] Testing nlags={nlags}, hid_dim={hid_dim}")

                zt_loss_list = []
                for z in range(nlatents):
                    print(f"      [latent] Training latent z{z}...")
                    z_loss = float("inf")
                    for run in range(args.runs):
                        model = SimpleARModel(nlags=nlags, in_dim=1, hid_dim=hid_dim)
                        _best_loss = train(model, zt[:, z, :])
                        z_loss = min(z_loss, _best_loss)
                        print(f"        [run {run+1}/{args.runs}] loss={_best_loss:.6f}")
                    zt_loss_list.append(z_loss)

                print(f"    [loop] Final loss list for nlags={nlags}, hid_dim={hid_dim}: {zt_loss_list}")
                ax.plot(zt_loss_list, alpha=0.5, marker="o")
                ax.set_xticks(list(range(nlatents)))
                ax.set_xlabel("Latents Index")
                ax.set_ylabel("Loss")

        f.suptitle(filename)
        os.makedirs(args.res_dir, exist_ok=True)
        out_path = os.path.join(args.res_dir, f"{filename.replace('.npz', '')}.png")
        f.savefig(out_path)
        print(f"[main] Saved figure to {out_path}")


if __name__ == "__main__":
    main()
