import argparse

import torch

from datasets.sim_dataset import StationaryDataset
from modules.instantaneous2 import InstantaneousProcess
from tools.gen_data import stationary_z_link
from tools.utils import load_yaml


def build_model(cfg):
    return InstantaneousProcess(
        input_dim=cfg["VAE"]["INPUT_DIM"],
        z_dim=cfg["VAE"]["LATENT_DIM"],
        z_dim_fix=cfg["VAE"]["LATENT_DIM_FIX"],
        z_dim_change=cfg["VAE"]["LATENT_DIM_CHANGE"],
        lag=cfg["VAE"]["LAG"],
        nclass=cfg["VAE"]["NCLASS"],
        hidden_dim=cfg["VAE"]["ENC"]["HIDDEN_DIM"],
        embedding_dim=cfg["VAE"]["EMBED_DIM"],
        lr=cfg["VAE"]["LR"],
        beta=cfg["VAE"]["BETA"],
        gamma=cfg["VAE"]["GAMMA"],
        theta=cfg["VAE"]["THETA"],
        decoder_dist=cfg["VAE"]["DEC"]["DIST"],
        correlation=cfg["MCC"]["CORR"],
        enable_flexible_sparsity=cfg["VAE"]["FLEXIBLE_SPARTSITY"]["ENABLE"],
        w_hist=(cfg["VAE"]["FLEXIBLE_SPARTSITY"]["HIST"]
                if cfg["VAE"]["FLEXIBLE_SPARTSITY"]["ENABLE"] else None),
        w_inst=(cfg["VAE"]["FLEXIBLE_SPARTSITY"]["INST"]
                if cfg["VAE"]["FLEXIBLE_SPARTSITY"]["ENABLE"] else None),
        layer_dims=cfg["VAE"].get("LAYER_DIMS"),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate data and run a minimal train-step smoke test.")
    parser.add_argument("--config", default="configs/I.yaml")
    parser.add_argument("--dataset", default="smoke_test")
    parser.add_argument("--seed", type=int, default=770)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--length", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cond-samples", type=int, default=128)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    z_dim = int(cfg["VAE"]["LATENT_DIM"])
    layer_dims = cfg["VAE"].get("LAYER_DIMS") or [z_dim]

    stationary_z_link(
        z_dim=z_dim,
        hist=0,
        inst=False,
        name=args.dataset,
        layer_dims=layer_dims,
        seed=args.seed,
        length=args.length,
        batch_size=args.samples,
        n_condition_samples=args.cond_samples,
    )

    data = StationaryDataset(dataset=args.dataset)
    batch_items = [data[i] for i in range(args.batch_size)]
    batch = {key: torch.stack([item[key] for item in batch_items])
             for key in batch_items[0]}

    model = build_model(cfg)
    model.train()
    x, x_recon, mus, logvars, zs, embeddings = model.forward(batch)
    sparsity_loss, recon_loss, kld_normal, kld_future = model.loss_function(
        x, x_recon, mus, logvars, zs, embeddings)
    loss = recon_loss + model.beta * kld_normal + model.gamma * kld_future + model.theta * sparsity_loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite smoke-test loss: {loss}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["VAE"]["LR"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print({
        "dataset": args.dataset,
        "dataset_len": len(data),
        "batch_xt_shape": tuple(batch["xt"].shape),
        "loss": float(loss.detach()),
    })


if __name__ == "__main__":
    main()
