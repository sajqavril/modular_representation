import os
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Tuple

EpsDist = Literal["gaussian", "laplace", "uniform"]
TransitionType = Literal["linear", "nonlinear"]
MixingType = Literal["linear", "nonlinear"]
Nonlinearity = Literal["tanh"]


@dataclass
class TemporalProcessConfig:
    # structure
    n_layers: int = 2
    n_vars_per_layer: int = 1
    d: Optional[int] = None  # observed dim; default = latent dim

    # temporal setup
    L: int = 1
    T: int = 10000
    burn_in: int = 300
    seed: Optional[int] = 0

    # latent dynamics
    transition_type: TransitionType = "linear"
    transition_nonlinearity: Nonlinearity = "tanh"
    B_density: float = 0.5
    B_scale: float = 0.2
    stable: bool = True
    stability_target: float = 0.95

    # observation mixing
    mixing_type: MixingType = "linear"

    # noise
    eps_dist: EpsDist = "gaussian"
    eps_params: Optional[Dict[str, Any]] = None

    @property
    def n(self) -> int:
        return self.n_layers * self.n_vars_per_layer

    @property
    def x_dim(self) -> int:
        return self.n if self.d is None else self.d


def _as_vec(param, n: int, name: str) -> np.ndarray:
    arr = np.asarray(param, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name} must be scalar or shape ({n},), got {arr.shape}")
    return arr


def sample_eps_zero_mean(
    T: int,
    n: int,
    dist: EpsDist,
    params: Optional[Dict[str, Any]],
    rng: np.random.Generator,
) -> np.ndarray:
    if params is None:
        params = {}

    if dist == "gaussian":
        sigma = _as_vec(params.get("sigma", 1.0), n, "sigma")
        if np.any(sigma <= 0):
            raise ValueError("sigma must be > 0.")
        return rng.normal(0.0, sigma, size=(T, n))

    if dist == "laplace":
        b = _as_vec(params.get("b", 1.0), n, "b")
        if np.any(b <= 0):
            raise ValueError("b must be > 0.")
        return rng.laplace(0.0, b, size=(T, n))

    if dist == "uniform":
        a = _as_vec(params.get("a", 1.0), n, "a")
        if np.any(a <= 0):
            raise ValueError("a must be > 0.")
        return rng.uniform(-a, a, size=(T, n))

    raise ValueError(f"Unknown dist: {dist}")


def apply_nonlinearity(x: np.ndarray, kind: Nonlinearity) -> np.ndarray:
    if kind == "tanh":
        return np.tanh(x)
    if kind == "relu":
        return np.maximum(0, x)
    if kind == "leakyrelu":
        return np.where(x > 0, x, 0.01 * x)
    raise ValueError(f"Unknown nonlinearity: {kind}")


def spectral_radius(M: np.ndarray) -> float:
    eigvals = np.linalg.eigvals(M)
    return float(np.max(np.abs(eigvals)))


def companion_matrix(B_list: np.ndarray) -> np.ndarray:
    L, n, _ = B_list.shape
    big = np.zeros((L * n, L * n), dtype=float)
    big[:n, :L * n] = np.concatenate([B_list[k] for k in range(L)], axis=1)
    if L > 1:
        big[n:, :-n] = np.eye(n * (L - 1))
    return big


def enforce_stability(
    B_list: np.ndarray,
    target: float = 0.90,
    max_iter: int = 100,
) -> Tuple[np.ndarray, float]:
    for _ in range(max_iter):
        rho = spectral_radius(companion_matrix(B_list))
        if not np.isfinite(rho):
            return B_list, rho
        if rho < target:
            return B_list, rho
        B_list = B_list * (target / rho)
    rho = spectral_radius(companion_matrix(B_list))
    return B_list, rho


def make_layered_transition(
    n_layers: int,
    n_vars_per_layer: int,
    density: float,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build B with:
      - sparsity (density)
      - block lower-triangular (layer constraint)
      - always allow diagonal (i == j at variable level)
    """
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be in [0, 1].")
    if scale < 0:
        raise ValueError("scale must be >= 0.")

    d = n_layers * n_vars_per_layer

    # 1) random weights
    mag = rng.uniform(0.6, 0.8, size=(d, d))
    sign = rng.choice([-1.0, 1.0], size=(d, d))
    W = sign * mag

    # 2) sparsity mask
    S = (rng.random((d, d)) < density).astype(float)

    # 3) layer indices for each variable
    layer_idx = np.repeat(np.arange(n_layers), n_vars_per_layer)

    # 4) block lower-triangular mask (j <= i)
    Lmask = (layer_idx[:, None] >= layer_idx[None, :]).astype(float)

    # 5) combine masks
    B = scale * W * S * Lmask

    # 6) always allow diagonal (override sparsity if needed)
    diag_mag = rng.uniform(0.6, 1.5, size=d)
    diag_sign = rng.choice([-1.0, 1.0], size=d)
    diag_vals = scale * diag_mag * diag_sign
    np.fill_diagonal(B, diag_vals)

    return B


def make_mixing_matrix(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(n, d))
    norms = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    return A / norms


def generate_process(
    cfg: TemporalProcessConfig,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Linear:
        z_t = sum_{tau=1..L} B_tau z_{t-tau} + eps_t

    Nonlinear:
        z_t = sum_{tau=1..L} W2_tau tanh(W1_tau z_{t-tau}) + eps_t

    x_t = z_t  or  A z_t

    Returns:
        z: (T, n)
        x: (T, d)
        B_list: (L, n, n) for linear transition, else None
        A: (d, n) or None
    """
    rng = np.random.default_rng(cfg.seed)
    n, d = cfg.n, cfg.x_dim

    B_list = None
    W1_list = None
    W2_list = None

    if cfg.transition_type == "linear":
        B_list = np.stack(
            [
                make_layered_transition(
                    n_layers=cfg.n_layers,
                    n_vars_per_layer=cfg.n_vars_per_layer,
                    density=cfg.B_density,
                    scale=cfg.B_scale,
                    rng=rng
                )
                for _ in range(cfg.L)
            ],
            axis=0,
        )

        if cfg.stable:
            B_list, rho = enforce_stability(B_list, target=cfg.stability_target)
            if not (np.isfinite(rho) and rho < cfg.stability_target):
                raise RuntimeError("Could not stabilize B_list.")
    else:
        # minimal-change nonlinear transition: two-layer tanh MLP per lag
        W1_list = np.stack(
            [
                make_layered_transition(
                    n_layers=cfg.n_layers,
                    n_vars_per_layer=cfg.n_vars_per_layer,
                    density=cfg.B_density,
                    scale=cfg.B_scale,
                    rng=rng
                )
                for _ in range(cfg.L)
            ],
            axis=0,
        )
        W2_list = np.stack(
            [
                make_layered_transition(
                    n_layers=cfg.n_layers,
                    n_vars_per_layer=cfg.n_vars_per_layer,
                    density=cfg.B_density,
                    scale=cfg.B_scale,
                    rng=rng
                )
                for _ in range(cfg.L)
            ],
            axis=0,
        )
        # out_lin1_list = [rng.random((2, 2)) for _ in range(n)]
        # out_lin2_list = [rng.random((1, 2)) for _ in range(n)]

    total = cfg.T + cfg.burn_in
    eps = sample_eps_zero_mean(total, n, cfg.eps_dist, cfg.eps_params, rng)

    z = np.zeros((total, n), dtype=float)
    for t in range(cfg.L):
        z[t] = eps[t]

    lagged_values = []
    for t in range(cfg.L, total):
        agg = np.zeros(n, dtype=float) # aggregated effects from all lags
        for tau in range(1, cfg.L + 1): # effects from each lag
            _prev_tau = z[t - tau]
            if cfg.transition_type == "linear":
                agg += B_list[tau - 1] @ _prev_tau
            else:
                lagged_values.append(_prev_tau)
                _hidden = apply_nonlinearity(W1_list[tau - 1] @ _prev_tau, "tanh")
                agg += W2_list[tau - 1] @ _hidden
        
        if cfg.transition_type == 'linear':
            z[t] = agg + eps[t]
        else:
            hist_avg = np.mean(np.stack(lagged_values, axis=0))
            z[t] = agg + hist_avg * eps[t]

            # for z_idx in range(n):
            #     _hidden = apply_nonlinearity(out_lin1_list[z_idx] @ np.array([agg[z_idx], eps[t,z_idx]]), "tanh")
            #     z[t,z_idx] = float(out_lin2_list[z_idx] @ _hidden)

        # avg_lagged = np.mean(np.concatenate(lagged_vals, axis=0))
        # noise_t = eps[t] * avg_lagged
        # z[t] = acc + noise_t

    z = z[cfg.burn_in:]

    A = None
    if cfg.mixing_type == "linear":
        A = make_mixing_matrix(d, n, rng)
        x = z @ A.T
    elif cfg.mixing_type == "nonlinear":
        lin1 = make_mixing_matrix(d // 2, n, rng)
        lin2 = make_mixing_matrix(d, d // 2, rng)
        _x = apply_nonlinearity(z @ lin1.T, kind="leakyrelu")
        x = _x @ lin2.T
    else:
        raise ValueError(f"Unknown mixing_type: {cfg.mixing_type}")

    return z, x, B_list, A


def generate_multi_trajectory_process(
    cfg: TemporalProcessConfig,
    num_traj: int = 64,
    T_each: int = 2000,
    reseed_each: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Shared transition and A across trajectories, independent noise per trajectory.
    For nonlinear transition, B_list is None.
    """
    rng_shared = np.random.default_rng(cfg.seed)
    n, d = cfg.n, cfg.x_dim

    B_list = None
    W1_list = None
    W2_list = None

    if cfg.transition_type == "linear":
        B_list = np.stack(
            [
                make_layered_transition(
                    n_layers=cfg.n_layers,
                    n_vars_per_layer=cfg.n_vars_per_layer,
                    density=cfg.B_density,
                    scale=cfg.B_scale,
                    rng=rng_shared
                )
                for _ in range(cfg.L)
            ],
            axis=0,
        )

        if cfg.stable:
            B_list, rho = enforce_stability(B_list, target=cfg.stability_target)
            if not (np.isfinite(rho) and rho < cfg.stability_target):
                raise RuntimeError("Could not stabilize shared B_list.")
    else:
        W1_list = np.stack(
            [
                make_layered_transition(
                    n_layers=cfg.n_layers,
                    n_vars_per_layer=cfg.n_vars_per_layer,
                    density=cfg.B_density,
                    scale=cfg.B_scale,
                    rng=rng_shared
                )
                for _ in range(cfg.L)
            ],
            axis=0,
        )
        W2_list = np.stack(
            [
                make_layered_transition(
                    n_layers=cfg.n_layers,
                    n_vars_per_layer=cfg.n_vars_per_layer,
                    density=cfg.B_density,
                    scale=cfg.B_scale,
                    rng=rng_shared
                )
                for _ in range(cfg.L)
            ],
            axis=0,
        )

    A = None
    lin1 = None
    lin2 = None
    if cfg.mixing_type == "linear":
        A = make_mixing_matrix(d, n, rng_shared)
    elif cfg.mixing_type == "nonlinear":
        # Share one nonlinear observation map across all trajectories.
        lin1 = make_mixing_matrix(d // 2, n, rng_shared)
        lin2 = make_mixing_matrix(d, d // 2, rng_shared)

    z_all = []
    x_all = []
    traj_ids = []

    base_seed = cfg.seed if cfg.seed is not None else 0

    for k in range(num_traj):
        seed_k = base_seed + 1000 * k if reseed_each else base_seed
        rng = np.random.default_rng(seed_k)

        total = T_each + cfg.burn_in
        eps = sample_eps_zero_mean(total, n, cfg.eps_dist, cfg.eps_params, rng)
        z = np.zeros((total, n), dtype=float)

        for t in range(min(cfg.L, total)):
            z[t] = eps[t]

        for t in range(cfg.L, total):
            agg = np.zeros(n, dtype=float)

            if cfg.transition_type == "nonlinear":
                lagged_values = []

            for tau in range(1, cfg.L + 1):
                prev = z[t - tau]
                if cfg.transition_type == "linear":
                    agg += B_list[tau - 1] @ prev
                else:
                    lagged_values.append(prev)
                    hidden = apply_nonlinearity(W1_list[tau - 1] @ prev, "tanh")
                    agg += W2_list[tau - 1] @ hidden

            if cfg.transition_type == "linear":
                z[t] = agg + eps[t]
            else:
                hist_avg = np.mean(np.stack(lagged_values, axis=0))
                z[t] = agg + hist_avg * eps[t]

        z = z[cfg.burn_in:]

        if cfg.mixing_type == "identity":
            x = z.copy()
        elif cfg.mixing_type == "linear":
            x = z @ A.T
        elif cfg.mixing_type == "nonlinear":
            _x = apply_nonlinearity(z @ lin1.T, kind="leakyrelu")
            x = _x @ lin2.T
        else:
            raise ValueError(f"Unknown mixing_type: {cfg.mixing_type}")

        z_all.append(z)
        x_all.append(x)
        traj_ids.append(np.full(T_each, k, dtype=int))

    return (
        np.stack(z_all, axis=0),
        np.stack(x_all, axis=0),
        B_list,
        A,
        np.concatenate(traj_ids, axis=0),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate temporal synthetic data from a preset.")
    parser.add_argument(
        "--preset",
        type=str,
        default="default_linear",
        help="Preset name defined in generation_presets.py",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Generate multiple trajectories instead of a single trajectory.",
    )
    parser.add_argument(
        "--num_traj",
        type=int,
        default=8,
        help="Number of trajectories when --multi is used.",
    )
    parser.add_argument(
        "--T_each",
        type=int,
        default=2000,
        help="Length of each trajectory when --multi is used.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional .npz output path.",
    )
    args = parser.parse_args()

    try:
        from synthetic.data.temporal_cfg import PRESET_CONFIGS
    except ModuleNotFoundError:
        from temporal_cfg import PRESET_CONFIGS

    if args.preset not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown preset '{args.preset}'. Available presets: {list(PRESET_CONFIGS.keys())}"
        )

    cfg = PRESET_CONFIGS[args.preset]

    print(f"Using preset: {args.preset}")
    print(cfg)

    if args.multi:
        z, x, B_list, A, traj_id = generate_multi_trajectory_process(
            cfg,
            num_traj=args.num_traj,
            T_each=args.T_each,
        )
        print("multi-trajectory mode")
        print("z shape:", z.shape)
        print("x shape:", x.shape)
        print("traj_id shape:", traj_id.shape)
    else:
        z, x, B_list, A = generate_process(cfg)
        traj_id = None
        print("single-trajectory mode")
        print("z shape:", z.shape)
        print("x shape:", x.shape)

    if B_list is not None:
        rho = spectral_radius(companion_matrix(B_list))
        print("companion spectral radius:", rho)
        print("B_list shape:", B_list.shape)
    else:
        print("nonlinear transition: no B_list to report/save")

    print("A shape:", None if A is None else A.shape)

    if args.save is not None:
        save_path = args.save

        # add prefix for multi-trajectory
        if args.multi:
            dirname, filename = os.path.split(save_path)
            save_path = os.path.join(dirname, f"multi_{filename}")

        save_dict = {
            "xt": x,
            "zt": z,
        }
        if B_list is not None:
            save_dict["B_list"] = B_list
        if A is not None:
            save_dict["A"] = A
        if traj_id is not None:
            save_dict["traj_id"] = traj_id

        np.savez(save_path, **save_dict)
        print(f"Saved to: {save_path}")
