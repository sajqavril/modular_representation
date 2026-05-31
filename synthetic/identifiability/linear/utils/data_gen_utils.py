from typing import Dict, List, Tuple, Optional
import numpy as np


def create_mixing_matrix(n, condition_number_max=2):
    """Create well-conditioned invertible matrix"""
    while True:
        A = np.random.randn(n, n)
        Q, R = np.linalg.qr(A)  # QR decomposition gives orthogonal Q
        
        # Make it well-conditioned
        D = np.diag(np.linspace(1, condition_number_max, n))
        W = Q @ D @ Q.T
        
        if np.linalg.cond(W) < condition_number_max:
            return W
        

def data_2_time_series(data, time_period):
    """
    Convert (n_sample, n_dim) -> (n_sample_new, time_period, n_dim)

    Parameters:
        data: np.ndarray of shape (n_sample, n_dim)
        time_period: int, length of each time window

    Returns:
        np.ndarray of shape (n_sample - time_period + 1, time_period, n_dim)
    """
    n_sample, n_dim = data.shape
    
    n_new = n_sample - time_period + 1
    if n_new <= 0:
        raise ValueError("time_period is larger than n_sample")

    result = np.zeros((n_new, time_period, n_dim))

    for i in range(n_new):
        result[i] = data[i:i + time_period]

    return result        


def _random_instantaneous_matrix(
    d: int, density: float = 0.5, scale: float = 0.3, rng=None
) -> np.ndarray:
    """
    Sample A^{[l]} as a STRICTLY LOWER TRIANGULAR matrix so that the
    within-layer instantaneous graph is a DAG and (I - A^{[l]}) is
    guaranteed invertible. This corresponds to (R1).
    """
    rng = rng if rng is not None else np.random.default_rng()
    A = rng.normal(0.0, scale, size=(d, d))
    A = np.tril(A, k=-1)                       # strictly lower triangular -> DAG
    mask = rng.random((d, d)) < density        # sparsify
    A = A * np.tril(mask, k=-1)
    return A


def _random_linear_matrix(
    d_out: int, d_in: int, scale: float = 0.3,
    spectral_radius: Optional[float] = 0.9, rng=None,
) -> np.ndarray:
    """
    Sample a generic linear map. Used for both within-layer transitions
    B^{[l]} and cross-layer time-lagged couplings C^{[l', l]}.
    If the matrix is square, optionally rescale to a target spectral
    radius < 1 so the recursion stays stable.
    """
    rng = rng if rng is not None else np.random.default_rng()
    W = rng.normal(0.0, scale, size=(d_out, d_in))
    if spectral_radius is not None and d_out == d_in:
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        if radius > 0:
            W = W * (spectral_radius / radius)
    return W


import numpy as np
from utils.config_utils import seed_everything

def sample_super_gaussian(n, mu=0.0, alpha=1.0, beta=1.0, seed=None):
    rng = np.random.default_rng(seed)

    # If Y ~ Gamma(1/beta, 1), then |X| = alpha * Y^(1/beta)
    y = rng.gamma(shape=1 / beta, scale=1.0, size=n)
    signs = rng.choice([-1.0, 1.0], size=n)

    return mu + alpha * signs * y ** (1 / beta)


def generate_hierarchical_synthetic_data(
    B_list,            # [B^{[1]}, ..., B^{[L]}]  within-layer transition matrices
    A_inst_list,       # [A^{[1]}, ..., A^{[L]}]  within-layer INSTANTANEOUS matrices
                       # (each MUST be strictly lower triangular -> DAG within a layer)
    H,                 # GLOBAL observation matrix, shape (x_dim, sum_l z_dim_l)
    C_dict=None,       # {(k, l): C^{[k,l]}}  lagged cross-layer matrices, k < l only
    num_samples=10240,
    length=1,
    noise_type="laplace",
    noise_scale=1.0,
    init_type="uniform",
):
    """
    Hierarchical latent time-series generator.

    Per-layer dynamics  (REQ: instantaneous + temporal WITHIN a layer):
        z^{[l]}_{t+1} = B^{[l]} z^{[l]}_t  +  A^{[l]} z^{[l]}_{t+1}  +  eps^{[l]}_{t+1}

    Cross-layer dynamics  (REQ: only TIME-LAGGED across layers, NO instantaneous):
        z^{[l]}_{t+1} += sum_{k in Pa(l), k < l}  C^{[k,l]} z^{[k]}_t

    Observation  (REQ: x_t is a linear mixing of the FULL latent at time t):
        z_t  := concat(z^{[1]}_t, ..., z^{[L]}_t)         in R^{sum_l z_dim_l}
        x_t  := H z_t                                      in R^{x_dim}

    Returns
    -------
    {
        "Z_layers": [Z^{[1]}, ..., Z^{[L]}],   each (num_samples, length+1, z_dim_l)
        "Z":        full concatenated latents, (num_samples, length+1, sum_l z_dim_l)
        "X":        observations,              (num_samples, length+1, x_dim)
    }
    """
    L = len(B_list)
    if C_dict is None:
        C_dict = {}

    # --------------------------------------------------------------------- #
    # Sanity checks                                                         #
    # --------------------------------------------------------------------- #
    z_dims = [B.shape[0] for B in B_list]
    z_total = sum(z_dims)

    for l in range(L):
        assert B_list[l].shape == (z_dims[l], z_dims[l]), \
            f"B^[{l}] must be square ({z_dims[l]}, {z_dims[l]})"
        assert A_inst_list[l].shape == (z_dims[l], z_dims[l]), \
            f"A^[{l}] must be square ({z_dims[l]}, {z_dims[l]})"
        # REQ: A^{[l]} encodes the within-layer instantaneous DAG, so it must
        # be strictly lower triangular (zero on/above the diagonal).
        assert np.allclose(np.triu(A_inst_list[l]), 0.0), \
            f"A^[{l}] must be strictly lower triangular (instantaneous DAG)."

    for (k, l), C in C_dict.items():
        # REQ: cross-layer influence is hierarchical and lagged only.
        # Parent k must be a *higher* layer (smaller index) than child l.
        assert k < l, (
            f"C^[{k},{l}] is invalid: cross-layer edges must go from a higher "
            f"layer to a lower one (k < l), and only with a time lag."
        )
        assert C.shape == (z_dims[l], z_dims[k]), \
            f"C^[{k},{l}] must have shape ({z_dims[l]}, {z_dims[k]})"

    # REQ: H mixes the FULL concatenated latent of the current time step.
    assert H.ndim == 2 and H.shape[1] == z_total, (
        f"H must have shape (x_dim, {z_total}) where {z_total} = sum of all "
        f"layer latent dims; got {H.shape}."
    )

    # --------------------------------------------------------------------- #
    # Initial latent states z^{[l]}_0 for each layer                        #
    # --------------------------------------------------------------------- #
    Z_layers = []
    for l in range(L):
        if init_type == "uniform":
            z0 = np.random.uniform(0.0, 1.0, (num_samples, z_dims[l]))
        elif init_type == "normal":
            z0 = np.random.normal(0.0, noise_scale, (num_samples, z_dims[l]))
        else:
            raise ValueError(f"Unsupported init_type: {init_type}")
        Z_layers.append([z0])  # list per layer; each entry is one time slice

    # --------------------------------------------------------------------- #
    # Roll the system forward                                               #
    # --------------------------------------------------------------------- #
    for t in range(length):
        # Compute z^{[l]}_{t+1} for every layer using ONLY time-t values of
        # other layers. This automatically enforces "no instantaneous
        # cross-layer dependence" (REQ).
        new_states = [None] * L

        for l in range(L):
            d_l = z_dims[l]
            z_prev_l = Z_layers[l][-1]                       # z^{[l]}_t

            # (1) REQ: within-layer historical term  B^{[l]} z^{[l]}_t
            z_hist = z_prev_l @ B_list[l].T

            # (2) REQ: lagged cross-layer term  sum_{k<l} C^{[k,l]} z^{[k]}_t
            cross = np.zeros_like(z_hist)
            for (k, ll), C in C_dict.items():
                if ll == l:
                    z_prev_k = Z_layers[k][-1]               # z^{[k]}_t (lagged!)
                    cross += z_prev_k @ C.T

            # (3) Innovation noise eps^{[l]}_{t+1}
            if noise_type == "laplace":
                noise = np.random.laplace(0.0, noise_scale, (num_samples, d_l))
            elif noise_type == "normal":
                noise = np.random.normal(0.0, noise_scale, (num_samples, d_l))
            elif noise_type == "uniform":
                noise = np.random.uniform(-noise_scale, noise_scale, (num_samples, d_l))
            elif noise_type =='sup-gauss':
                noise = np.zeros((num_samples, d_l))
                for i in range(d_l):
                    noise[:, i] = sample_super_gaussian(num_samples, mu=0.0, alpha=noise_scale, beta=0.5)
            else:
                raise ValueError(f"Unsupported noise_type: {noise_type}")

            # Everything except the within-layer instantaneous self-loop:
            base = z_hist + cross + noise

            # (4) REQ: within-layer instantaneous term  A^{[l]} z^{[l]}_{t+1}
            # Because A^{[l]} is strictly lower triangular, we can solve
            # (I - A^{[l]}) z^{[l]}_{t+1} = base   coordinate-by-coordinate
            # in a single forward sweep:
            #   z^{[l]}_{t+1, i} = base_i + sum_{j<i} A^{[l]}_{ij} z^{[l]}_{t+1, j}
            A_l = A_inst_list[l]
            z_next_l = np.zeros((num_samples, d_l))
            for i in range(d_l):
                z_next_l[:, i] = base[:, i]
                if i > 0:
                    z_next_l[:, i] += z_next_l[:, :i] @ A_l[i, :i]

            new_states[l] = z_next_l

        for l in range(L):
            Z_layers[l].append(new_states[l])

    # --------------------------------------------------------------------- #
    # Stack per-layer latents to (num_samples, length+1, z_dim_l)           #
    # --------------------------------------------------------------------- #
    Z_layers = [np.array(Z_layers[l]).transpose(1, 0, 2) for l in range(L)]

    # REQ: concatenate ALL layers' latents at every time step, then mix with
    # the GLOBAL H to produce a single observation x_t.
    Z_full = np.concatenate(Z_layers, axis=-1)               # (N, T+1, z_total)
    X = Z_full @ H.T                                         # (N, T+1, x_dim)

    return X, Z_full
