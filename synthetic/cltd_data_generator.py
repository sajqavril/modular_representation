import os
import torch
import random
import argparse
import numpy as np
from scipy.stats import ortho_group

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gen_masked_trans(nlatents, mask, cond_threshold):
    X = np.random.uniform(0, 2, (nlatents, nlatents)) - 1
    for i in range(nlatents):
        norm = np.sqrt((X[:, i] ** 2).sum())
        if norm > 0:
            X[:, i] /= norm

    while np.linalg.cond(X) > cond_threshold:
        X = np.random.uniform(0, 2, (nlatents, nlatents)) - 1
        for i in range(nlatents):
            norm = np.sqrt((X[:, i] ** 2).sum())
            if norm > 0:
                X[:, i] /= norm
    
    return X * mask


def gen_synthetic_data(nlags, 
                       nnlayer, 
                       nlatents, 
                       length, 
                       sample_size, 
                       M_sparsity, 
                       seed, 
                       noise_mode, 
                       neg_slope, 
                       noise_dist,
                       burn_in=50):
    # compute threshold for the conditional value
    print("compute threshold for the conditional value...")
    conds = []
    for _ in range(10_000):
        X = np.random.uniform(low=1, high=2, size=(nlatents, nlatents))
        for i in range(nlatents):
            X[:, i] /= np.sqrt((X[:, i] ** 2).sum())
        conds.append(np.linalg.cond(X))

    cond_threshold = np.percentile(conds, 25)

    # generate temporal Bs
    print("generate temporal Bs (restrict to the history from the same layer)...")
    Bs = []
    B_mask = np.eye(nlatents)
    for lag in range(nlags): 
        B = gen_masked_trans(nlatents, B_mask, cond_threshold)
        Bs.append(B)
    
    # generate instantaneous M
    print("generate instantaneous M (using a base mask to ensure the hierarchy)...")
    M_base_mask = np.tril(np.ones(shape=(nlatents, nlatents)), k=-1)
    M_structure_mask = (np.random.uniform(low=0, high=1, size=(nlatents, nlatents)) > M_sparsity).astype(float)
    M_mask = M_base_mask * M_structure_mask
    M = gen_masked_trans(nlatents, M_mask, cond_threshold)

    # generate mixing transitions
    print("generate mixing transitions...")
    As = []
    for layer in range(nnlayer):
        A = ortho_group.rvs(nlatents, random_state=seed)
        As.append(A)
    
    # initialize yt history
    print("initialize yt history...")
    if noise_dist == 'normal':
        yt = np.random.normal(loc=0, scale=1, size=(nlags, nlatents, sample_size))
    elif noise_dist == 'uniform':
        yt = np.random.uniform(low=0, high=1, size=(nlags, nlatents, sample_size))
    else: # laplace
        yt = np.random.laplace(loc=0, scale=1, size=(nlags, nlatents, sample_size))
    yt = (yt - np.mean(yt, axis=-1, keepdims=True)) / np.std(yt, axis=-1, keepdims=True)

    # main generation loop
    total_steps = length + burn_in
    print(f"Generate zt for {total_steps} steps (will discard first {burn_in})...")
    zt_all = []
    xt_all = []

    for t in range(total_steps):
        _zt = np.zeros((nlatents, sample_size))

        # time-lagged contributions
        for lag in range(nlags):
            _zt_lag = np.dot(Bs[lag], yt[lag])
            _zt += np.where(_zt_lag > 0, _zt_lag, _zt_lag * neg_slope)

        # instantaneous left-to-right
        for i in range(nlatents):
            # noise
            if noise_mode == 1:
                if noise_dist == 'normal':
                    noise_i = np.random.normal(loc=0, scale=1+0.1*(nlatents-i), size=(sample_size,))
                elif noise_dist == 'uniform':
                    noise_i = np.random.uniform(low=0, high=1+0.5*(nlatents-i), size=(sample_size,))
                else:
                    noise_i = np.random.laplace(loc=0, scale=1+0.1*(nlatents-i), size=(sample_size,))
            else:
                if noise_dist == 'normal':
                    noise_i = np.random.normal(loc=0, scale=1, size=(sample_size,))
                elif noise_dist == 'uniform':
                    noise_i = np.random.uniform(low=0, high=1, size=(sample_size,))
                else:
                    noise_i = np.random.laplace(loc=0, scale=1, size=(sample_size,))

            # instantaneous contrib
            inst_contrib = np.dot(M[i, :i], _zt[:i])
            z_i = _zt[i] + inst_contrib + noise_i
            z_i = np.where(z_i > 0, z_i, z_i * neg_slope)
            _zt[i] = z_i

        zt_all.append(_zt)

        # update yt buffer
        yt = np.concatenate([np.expand_dims(_zt, axis=0), yt[:nlags-1]], axis=0)

        # mixing layers
        _xt = _zt
        for layer in range(nnlayer):
            _xt = np.dot(As[layer], _xt)
            _xt = np.where(_xt > 0, _xt, _xt * neg_slope)
        xt_all.append(_xt)

    # keep only last `length` steps
    zt = np.stack(zt_all[-length:], axis=0)
    xt = np.stack(xt_all[-length:], axis=0)

    return (zt, xt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=221)
    parser.add_argument("--nlags", type=int, default=2)
    parser.add_argument("--nlatents", type=int, default=3)
    parser.add_argument("--length", type=int, default=5)
    # parser.add_argument("--B-sparsity", type=float, default=0.4)
    parser.add_argument("--M-sparsity", type=float, default=0.4)
    parser.add_argument("--noise-dist", type=str, default='uniform', choices=['normal', 'laplace', 'uniform'], help='set the distribution class for all the noise terms in the data model')
    parser.add_argument("--noise-mode", type=int, choices=[0,1], default=1, help='0=same variance for all noise terms; 1=highist variance for z1, grandually increases')
    parser.add_argument("--nnlayer", type=int, default=3, help='number of nn layers for mixing function')
    parser.add_argument("--neg-slope", type=float, default=0.2)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--save-dir", type=str, default='datasets')
    parser.add_argument("--burn-in", type=int, default=50, help="number of warm-up steps to discard")

    args = parser.parse_args()

    # set random seed
    set_seed(args.seed)

    zt, xt = gen_synthetic_data(nlags=args.nlags,
                                nlatents=args.nlatents,
                                length=args.length,
                                nnlayer=args.nnlayer,
                                sample_size=args.sample_size,
                                # B_sparsity=args.B_sparsity,
                                M_sparsity=args.M_sparsity,
                                seed=args.seed,
                                noise_dist=args.noise_dist,
                                noise_mode=args.noise_mode,
                                neg_slope=args.neg_slope,
                                burn_in=args.burn_in)
    
    save_dir = os.path.join('.', args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    args_dict = vars(args)
    parts = [f"{k}={v}" for k, v in sorted(args_dict.items())]
    save_name = f"data_{'_'.join(parts)}.npz"

    print(f"saving zt and xt to {save_dir}/{save_name}...")
    data = {
        "zt": zt,
        "xt": xt
    }
    np.savez(os.path.join(save_dir, save_name), **data)


if __name__ == "__main__":
    main()
    