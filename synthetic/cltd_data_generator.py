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
        X[:,i] /= np.sqrt((X[:, i] ** 2).sum())
    # X = X * mask

    while np.linalg.cond(X) > cond_threshold:
        X = np.random.uniform(0, 2, (nlatents, nlatents)) - 1
        for i in range(nlatents):
            X[:,i] /= np.sqrt((X[:, i] ** 2).sum())
        # X = X * mask
    
    return X * mask


def gen_synthetic_data(nlags, nnlayer, nlatents, length, sample_size, B_sparsity, M_sparsity, seed, noise_mode, neg_slope, noise_dist):
    # compute threshold for the conditional value
    print("compute threshold for the conditional value...")
    conds = []
    for _ in range(10_000):
        X = np.random.uniform(low=1, high=2, size=(nlatents, nlatents))
        for i in range(nlatents):
            X[:, i] /= np.sqrt((X[:, i] ** 2).sum())
        conds.append(np.linalg.cond(X))

    cond_threshold = np.percentile(conds, 25)

    # generate temporal Bs (using a base mask to ensure the hierarchy)
    print("generate temporal Bs (using a base mask to ensure the hierarchy)...")
    Bs = []
    B_base_mask = np.tril(np.ones(shape=(nlatents, nlatents)), k=0)
    B_structure_mask = (np.random.uniform(low=0, high=1, size=(nlatents, nlatents)) > B_sparsity).astype(float)
    B_mask = B_base_mask * B_structure_mask
    for lag in range(nlags): 
        B = gen_masked_trans(nlatents, B_mask, cond_threshold)
        Bs.append(B)
    
    # generate instantaneous M (using a base mask to ensure the hierarchy)
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
    
    # generate terms before the output zt, named by yt and in a reversed order
    print("generate terms before the output zt, named by yt and in a reversed order...")
    if noise_dist == 'normal':
        yt = np.random.normal(loc=0, scale=1, size=(nlags, nlatents, sample_size))
    elif noise_dist == 'uniform':
        yt = np.random.uniform(low=0, high=1, size=(nlags, nlatents, sample_size))
    else: # noise_dist == laplace
        yt = np.random.laplace(loc=0, scale=1, size=(nlags, nlatents, sample_size))
    yt = (yt - np.mean(yt, axis=-1, keepdims=True)) / np.std(yt, axis=-1, keepdims=True)

    # generate zt in left-to-right manner (top-down contraint has been implemented in B and M masks) by yt, Bs, and M
    print("generate zt in left-to-right manner (top-down contraint has been implemented in B and M masks) by yt, Bs, and M...")
    zt = []
    xt = []
    for t in range(length):
        # generate noise terms
        if noise_mode == 1:
            if noise_dist == 'normal':
                _noise_zt = [np.random.normal(loc=0, scale=1+0.1*(nlatents-z), size=(sample_size,)) for z in range(nlatents)]
            elif noise_dist == 'uniform':
                _noise_zt = [np.random.uniform(low=0, high=1+0.5*(nlatents-z), size=(sample_size,)) for z in range(nlatents)]
            else: # noise_dist == laplace
                _noise_zt = [np.random.laplace(loc=0, scale=1+0.1*(nlatents-z), size=(sample_size,)) for z in range(nlatents)]
            noise_zt = np.stack(_noise_zt, axis=0)
        else:
            if noise_dist == 'normal':
                noise_zt = np.random.normal(loc=0, scale=1, size=(nlatents, sample_size))
            elif noise_dist == 'uniform':
                noise_zt = np.random.uniform(low=0, high=1, size=(nlatents, sample_size))
            else: # noise_dist == laplace
                noise_zt = np.random.laplace(loc=0, scale=1, size=(nlatents, sample_size))
        
        # compute zt
        _zt = 0
        for lag in range(nlags): # time-laged
            _zt_lag = np.dot(Bs[lag], yt[lag])
            _zt += np.where(_zt_lag > 0, _zt_lag, _zt_lag * neg_slope)
        
        _zt_inst = np.dot(M, _zt) # instantaneous
        _zt += np.where(_zt_inst > 0, _zt_inst, _zt_inst * neg_slope)
        
        _zt = np.where(_zt > 0, _zt, _zt * neg_slope) + noise_zt # noise

        zt.append(_zt)
        yt = np.concatenate([np.expand_dims(_zt, axis=0), yt], axis=0)

        # generate xt by zt and As
        _xt = _zt
        for layer in range(nnlayer):
            _xt = np.dot(As[layer], _xt)
            _xt = np.where(_xt > 0, _xt, _xt * neg_slope)
        
        xt.append(_xt)
 
    zt = np.stack(zt, axis=0)
    xt = np.stack(xt, axis=0)

    return (zt, xt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=221)
    parser.add_argument("--nlags", type=int, default=2)
    parser.add_argument("--nlatents", type=int, default=3)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--B-sparsity", type=float, default=0.4)
    parser.add_argument("--M-sparsity", type=float, default=0.4)
    parser.add_argument("--noise-dist", type=str, default='uniform', choices=['normal', 'laplace', 'uniform'], help='set the distribution class for all the noise terms in the data model')
    parser.add_argument("--noise-mode", type=int, choices=[0,1], default=1, help='0=same variance for all noise terms; 1=highist variance for z1, grandually increases')
    parser.add_argument("--nnlayer", type=int, default=3, help='number of nn layers for mixing function')
    parser.add_argument("--neg-slope", type=float, default=0.2)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--save-dir", type=str, default='datasets')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)

    # set random seed
    set_seed(args.seed)

    zt, xt = gen_synthetic_data(nlags=args.nlags,
                                nlatents=args.nlatents,
                                length=args.length,
                                nnlayer=args.nnlayer,
                                sample_size=args.sample_size,
                                B_sparsity=args.B_sparsity,
                                M_sparsity=args.M_sparsity,
                                seed=args.seed,
                                noise_dist=args.noise_dist,
                                noise_mode=args.noise_mode,
                                neg_slope=args.neg_slope)
    
    save_dir = os.path.join('.', args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    args_dict = vars(args)
    parts = [f"{k}={v}" for k, v in sorted(args_dict.items())]
    zt_name = f"{'_'.join(parts)}_zt"
    xt_name = f"{'_'.join(parts)}_xt"

    print(f"saving zt and xt to {save_dir}...")
    np.save(os.path.join(save_dir, zt_name), zt.to('cpu'))
    np.save(os.path.join(save_dir, xt_name), xt.to('cpu'))


if __name__ == "__main__":
    main()
    