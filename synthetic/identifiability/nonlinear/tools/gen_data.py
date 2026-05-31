import argparse
import os
import glob
import tqdm
import torch
import scipy
import random
import numpy as np
import yaml
from torch import nn
from torch.nn import init
from collections import deque
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import ortho_group
from sklearn.preprocessing import scale
# from IDOL.tools.utils import create_sparse_transitions, controlable_sparse_transitions

VALIDATION_RATIO = 0.2
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'dataset'))
standard_scaler = preprocessing.StandardScaler()

def leaky_ReLU_1d(d, negSlope):
    if d > 0:
        return d
    else:
        return d * negSlope

leaky1d = np.vectorize(leaky_ReLU_1d)

def leaky_ReLU(D, negSlope):
    assert negSlope > 0
    return leaky1d(D, negSlope)

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()

def sigmoidAct(x):
    return 1. / (1 + np.exp(-1 * x))

def generateUniformMat(Ncomp, condT):
    """
    generate a random matrix by sampling each element uniformly at random
    check condition number versus a condition threshold
    """
    A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
    for i in range(Ncomp):
        A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    while np.linalg.cond(A) > condT:
        # generate a new A matrix!
        A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
        for i in range(Ncomp):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    return A


def block_lower_triangular_mask(layer_dims):
    """
    Build a mask for the transition matrix `A` used as `y_t = y_{t-1} @ A`.
 
    In `A` (the matrix used in the code, where rows index the SOURCE / "from"
    component and columns index the DESTINATION / "to" component), the block
    at (source_layer = i, destination_layer = j) is allowed to be non-zero
    iff i <= j (a source layer can only feed itself or *later* layers).
 
    Equivalently, written in the standard math notation y_t = A_std @ y_{t-1}
    (where rows = "to", cols = "from"), the matrix is BLOCK LOWER TRIANGULAR:
        A_std[layer_i_rows, layer_j_cols] = 0  whenever  j > i
    i.e. layer i is only influenced by layers 1..i.
 
    For two layers with dims [d1, d2] this produces (in standard notation):
        [ L1 -> L1 ,    0       ]
        [ L1 -> L2 ,  L2 -> L2  ]
    which matches the user-requested structure (the (L2 -> L1) block is 0).
    """
    n_layers = len(layer_dims)
    total = sum(layer_dims)
    mask = np.zeros((total, total), dtype=np.float32)
    bounds = np.cumsum([0] + list(layer_dims))
    for src in range(n_layers):
        for dst in range(src, n_layers):
            mask[bounds[src]:bounds[src + 1],
                 bounds[dst]:bounds[dst + 1]] = 1.0
    return mask
 
 
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stationary_z_link(z_dim = 5, hist = 0, inst = False, name = 'data',
                      layer_dims = None, seed = None, length = 5,
                      batch_size = 10000, n_condition_samples = 10000):
    if seed is not None:
        setup_seed(seed)
    lags = 1
    Nlayer = 2
    length = int(length)
    condList = []
    negSlope = 0.2
    latent_size = z_dim
    transitions = []
    noise_scale = 0.1
    batch_size = int(batch_size)
    Niter4condThresh = int(n_condition_samples)
 
    path = os.path.join(root_dir, name)
    os.makedirs(path, exist_ok=True)
 
    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))
 
    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
 
    # Build the block-lower-triangular mask once; it is shared across lags.
    if layer_dims is None:
        # back-compat: a single layer covering the full latent space
        layer_dims = [latent_size]
    assert sum(layer_dims) == latent_size, \
        "sum(layer_dims) (%d) must equal z_dim/latent_size (%d)" % (sum(layer_dims), latent_size)
    layer_mask = block_lower_triangular_mask(layer_dims)
 
    for l in range(lags):
 
        if hist == 0:
            B = generateUniformMat(latent_size, condThresh)
            # Enforce the block-lower-triangular layer structure.
            # In `A` (used as y_t = y_{t-1} @ A), this is block-UPPER-triangular:
            # source layer i -> destination layer j is allowed only if i <= j,
            # which corresponds to the user-requested block-LOWER-triangular
            # matrix in the standard "rows=to, cols=from" notation.
            B = B * layer_mask
            transitions.append(B)
        
    transitions.reverse()
 
    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)
 
    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
 
    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
        
    # Mixing function
    for i in range(length):
        # Transition function
        y_t_noise = np.random.normal(0, noise_scale, (batch_size, latent_size)) * 2
        # Modulate the noise scale with averaged history
        # y_t_noise = y_t_noise * np.mean(y_l, axis=1)
        y_t = 0
        # transition
        for l in range(lags):
            y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
        y_t = leaky_ReLU(y_t, negSlope)
 
 
        p_hist = 0.2 # the weight of history data
        y_t[:,0] = y_t[:,0] * y_t_noise[:,0] + y_t_noise[:,0]
        for i in range(1, latent_size):
            if inst:
                y_t[:,i] = (p_hist * y_t[:,i] + (1-p_hist) * y_t[:,i-1]) * y_t_noise[:,i] + y_t_noise[:,i]
            else:
                y_t[:,i] = y_t[:,i] * y_t_noise[:,i] + y_t_noise[:,i]                
 
 
        yt.append(y_t)
        # Mixing function
        mixedDat = np.copy(y_t)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_t = np.copy(mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
 
    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)
 
    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)
 
    # Persist the layer partition so the training side can pick it up.
    np.save(os.path.join(path, "layer_dims"), np.array(layer_dims, dtype=np.int64))
 
    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)    
 

def parse_layer_dims(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(dim) for dim in value]
    return [int(dim) for dim in value.replace(",", " " ).split()]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a synthetic stationary dataset.")
    parser.add_argument("--config", default=None, help="YAML config used to infer DATASET, LATENT_DIM, and LAYER_DIMS.")
    parser.add_argument("--name", default=None, help="Output dataset name under datasets/dataset/. Defaults to config DATASET.")
    parser.add_argument("--z-dim", type=int, default=None)
    parser.add_argument("--layer-dims", default=None, help="Comma- or space-separated latent layer dimensions, e.g. '2,2'.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hist", type=int, default=0)
    parser.add_argument("--inst", action="store_true")
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--cond-samples", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = {}
    if args.config is not None:
        with open(args.config, "r") as stream:
            cfg = yaml.safe_load(stream)

    z_dim = args.z_dim
    if z_dim is None:
        z_dim = int(cfg.get("VAE", {}).get("LATENT_DIM", 4))

    layer_dims = parse_layer_dims(args.layer_dims)
    if layer_dims is None:
        layer_dims = parse_layer_dims(cfg.get("VAE", {}).get("LAYER_DIMS"))
    if layer_dims is None:
        layer_dims = [z_dim]

    name = args.name
    if name is None:
        name = cfg.get("DATASET", "I")

    stationary_z_link(z_dim=z_dim, hist=args.hist, inst=args.inst,
                      name=name, layer_dims=layer_dims, seed=args.seed,
                      length=args.length, batch_size=args.batch_size,
                      n_condition_samples=args.cond_samples)
    print("Generated dataset '%s' under %s" % (name, root_dir))


if __name__ == "__main__":
    main()