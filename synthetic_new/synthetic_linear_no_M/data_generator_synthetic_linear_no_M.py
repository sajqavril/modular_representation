import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Tuple


EpsDist = Literal["gaussian", "laplace", "uniform"]


@dataclass
class LinearTemporalProcessConfig:
    d: int                    # dimension of z_t and x_t (same)
    L: int                    # lag order
    T: int                    # number of samples to generate (t=0..T-1 returned)
    burn_in: int = 200        # extra steps discarded to reduce dependence on init
    seed: Optional[int] = None

    # B_tau generation
    B_density: float = 0.2    # probability a strictly-lower-triangular entry is nonzero
    B_scale: float = 0.05     # magnitude scale for nonzero B entries
    stable: bool = True       # attempt to scale B's to be stable-ish

    # epsilon
    eps_dist: EpsDist = "gaussian"
    # parameters can be scalars or length-d arrays; same across time
    eps_params: Dict[str, Any] = None  # see sample_eps() docs below

    # x = A z
    A: Optional[np.ndarray] = None      # if None -> identity

def _as_vec(param, d: int, name: str) -> np.ndarray:
    arr = np.asarray(param, dtype=float)
    if arr.ndim == 0:
        return np.full(d, float(arr))
    if arr.shape != (d,):
        raise ValueError(f"{name} must be a scalar or shape (d,), got {arr.shape}")
    return arr

def sample_eps_zero_mean(
    T: int,
    d: int,
    dist: EpsDist,
    params: Dict[str, Any],
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate eps_t with:
      - temporal whiteness: i.i.d. over time
      - mutual uncorrelatedness across dims: independent sampling per dim
      - ZERO MEAN for each dimension

    params by dist:
      - gaussian: {"sigma": scalar/(d,)}                    # mean forced to 0
      - laplace : {"b": scalar/(d,)}                        # mean forced to 0
      - uniform : either:
          (A) {"a": scalar/(d,)}  -> U[-a_i, a_i] (mean 0), OR
          (B) {"low": scalar/(d,), "high": scalar/(d,)} -> samples are recentered to mean 0
    """
    if params is None:
        params = {}

    if dist == "gaussian":
        sigma = _as_vec(params.get("sigma", 1.0), d, "sigma")
        if np.any(sigma <= 0):
            raise ValueError("All sigma must be > 0.")
        return rng.normal(loc=0.0, scale=sigma, size=(T, d))

    if dist == "laplace":
        b = _as_vec(params.get("b", 1.0), d, "b")
        if np.any(b <= 0):
            raise ValueError("All b must be > 0.")
        return rng.laplace(loc=0.0, scale=b, size=(T, d))

    if dist == "uniform":
        if "a" in params:
            a = _as_vec(params["a"], d, "a")
            if np.any(a <= 0):
                raise ValueError("All a must be > 0.")
            return rng.uniform(low=-a, high=a, size=(T, d))

        low = _as_vec(params.get("low", -1.0), d, "low")
        high = _as_vec(params.get("high", 1.0), d, "high")
        if np.any(high <= low):
            raise ValueError("All high must be > low (elementwise).")
        eps = rng.uniform(low=low, high=high, size=(T, d))
        center = (low + high) / 2.0
        return eps - center  # enforce exact zero mean parameter-wise

    raise ValueError(f"Unknown dist: {dist}")


def make_lower_triangular_sparse_matrix(
    d: int,
    density: float,
    scale: float,
    rng: np.random.Generator,
    dist: Literal["normal", "uniform"] = "normal",
) -> np.ndarray:
    """
    Create a dxd lower-triangular matrix (including diagonal) where entries
    above diagonal are zero, and entries on/under diagonal are sparse.

    density applies to positions i>=j (lower triangle including diagonal):
      each such entry is nonzero with probability=density.

    Nonzero values are drawn then multiplied by `scale`.
    """
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be in [0,1].")
    if scale < 0:
        raise ValueError("scale must be >= 0.")

    mask = rng.random((d, d)) < density
    tril_mask = np.tril(mask).astype(float)

    if dist == "normal":
        vals = rng.normal(size=(d, d))
    else:
        vals = rng.uniform(-1.0, 1.0, size=(d, d))

    B = scale * vals * tril_mask
    # ensure strictly upper triangle is zero
    B = np.tril(B)
    return B


def spectral_radius(M: np.ndarray) -> float:
    """Spectral radius (max abs eigenvalue)."""
    eigvals = np.linalg.eigvals(M)
    return float(np.max(np.abs(eigvals)))


def generate_process(cfg: LinearTemporalProcessConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate z_t and x_t from:
        z_t = sum_{tau=1..L} B_tau z_{t-tau} + eps_t
        x_t = A z_t

    Returns:
        z: (T, d)
        x: (T, d)
        B_list: (L, d, d)
        A: (d, d)
    """
    d, L, T = cfg.d, cfg.L, cfg.T
    if L < 1:
        raise ValueError("L must be >= 1.")
    if T < 1:
        raise ValueError("T must be >= 1.")

    rng = np.random.default_rng(cfg.seed)

    # A
    A = np.eye(d) if cfg.A is None else np.asarray(cfg.A, dtype=float)
    if A.shape != (d, d):
        raise ValueError(f"A must have shape {(d, d)}, got {A.shape}")

    # B_tau (same density across tau)
    B_list = np.stack([
        make_lower_triangular_sparse_matrix(d, cfg.B_density, cfg.B_scale, rng)
        for _ in range(L)
    ], axis=0)  # (L, d, d)

    # optional stability-ish scaling:
    # Convert VAR(L) to companion matrix and scale B's so companion spectral radius < 1
    if cfg.stable:
        # Build companion matrix C of size (d*L, d*L)
        # [B1 B2 ... BL]
        # [I  0  ... 0 ]
        # [0  I  ... 0 ]
        # ...
        big = np.zeros((d * L, d * L), dtype=float)
        big[:d, :d * L] = np.concatenate([B_list[t] for t in range(L)], axis=1)
        if L > 1:
            big[d:, :-d] = np.eye(d * (L - 1))
        rho = spectral_radius(big)
        # If rho is too large, scale down to target < 1
        target = 0.95
        if rho > 0 and rho >= target:
            factor = target / rho
            B_list *= factor

    # Simulate with burn-in
    burn = int(cfg.burn_in)
    total = T + burn
    eps = sample_eps_zero_mean(total, d, cfg.eps_dist, cfg.eps_params, rng)

    z = np.zeros((total, d), dtype=float)

    # initialize first L values with eps (or zeros). Here: eps init is fine.
    for t in range(min(L, total)):
        z[t] = eps[t]

    for t in range(L, total):
        acc = np.zeros(d, dtype=float)
        for tau in range(1, L + 1):
            acc += B_list[tau - 1] @ z[t - tau]
        z[t] = acc + eps[t]

    z_out = z[burn:]
    x_out = (z_out @ A.T)  # x_t = A z_t, using row vectors => multiply by A^T

    return z_out, x_out, B_list, A

# ---- Example usage ----
if __name__ == "__main__":
    # gaussian
    gaussian_cfg = LinearTemporalProcessConfig(
        d=3,
        L=1,
        T=10000,
        burn_in=300,
        seed=0,
        B_density=0.3,
        B_scale=0.08,
        stable=True,
        eps_dist="gaussian",
        eps_params={"sigma": np.array([1.0,1.0,1.0])},
        A=np.eye(3),
    )
    # uniform
    uniform_cfg = LinearTemporalProcessConfig(
        d=3,
        L=1,
        T=10000,
        burn_in=300,
        seed=0,
        B_density=0.3,
        B_scale=0.08,
        stable=True,
        eps_dist="uniform",
        eps_params={"a": np.array([1.0,1.0,1.0])},
        A=np.eye(3),
    )
    # laplace
    laplace_cfg = LinearTemporalProcessConfig(
        d=3,
        L=1,
        T=10000,
        burn_in=300,
        seed=0,
        B_density=0.3,
        B_scale=0.08,
        stable=True,
        eps_dist="laplace",
        eps_params={"b": np.array([1.0,1.0,1.0])},
        A=np.eye(3),
    )
    
    cfgs = [laplace_cfg, uniform_cfg, gaussian_cfg]
    names = ['laplace_3_same', 'uniform_3_same', 'gaussian_3_same']

    for cfg_idx in range(len(cfgs)):
        z, x, B_list, A = generate_process(cfgs[cfg_idx])
        print("z shape:", z.shape, "x shape:", x.shape)
        print("B_list shape:", B_list.shape)
        # x_views = generate_multiview_x_datasets(
        #     z=z,
        #     A=A,
        #     save_dir="./data/multiview_data",
        #     prefix="linear_var",
        #     save_format="npz",
        # )
        np.savez(file="./data/" + names[cfg_idx], xt=x, zt=z, B_list=B_list, A=A)
