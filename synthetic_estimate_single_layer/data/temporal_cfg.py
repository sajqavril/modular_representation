import numpy as np
from temporal_generator import TemporalProcessConfig

LINEAR_GAUSSIAN = TemporalProcessConfig(
    n_layers=2,
    n_vars_per_layer=1,
    d=None,  # defaults to n
    L=1,
    T=50000,
    burn_in=300,
    seed=0,
    transition_type="linear",
    mixing_type="linear",
    B_density=1.0,
    B_scale=0.5,
    stable=True,
    eps_dist="gaussian",
    eps_params={"sigma": 0.1},
)

LINEAR_UNIFORM = TemporalProcessConfig(
    n_layers=2,
    n_vars_per_layer=1,
    d=None,  # defaults to n
    L=1,
    T=50000,
    burn_in=300,
    seed=0,
    transition_type="linear",
    mixing_type="linear",
    B_density=1.0,
    B_scale=1.0,
    stable=True,
    eps_dist="uniform",
    eps_params={"sigma": 0.1},
)

LINEAR_LAPLACE = TemporalProcessConfig(
    n_layers=2,
    n_vars_per_layer=1,
    d=None,  # defaults to n
    L=1,
    T=50000,
    burn_in=300,
    seed=0,
    transition_type="linear",
    mixing_type="linear",
    B_density=1.0,
    B_scale=1.0,
    stable=True,
    eps_dist="laplace",
    eps_params={"sigma": 0.1},
)

NONLINEAR_TRANSITION = TemporalProcessConfig(
    n_layers=2,
    n_vars_per_layer=1,
    d=None,
    L=1,
    T=50000,
    burn_in=300,
    seed=0,
    transition_type="nonlinear",
    transition_nonlinearity="tanh",
    mixing_type="linear",
    B_density=1.0,
    B_scale=0.5,
    stable=True,
    eps_dist="gaussian",
    eps_params={"sigma": 0.1},
)

THREE_LAYER_TWO_VAR = TemporalProcessConfig(
    n_layers=3,
    n_vars_per_layer=2,
    d=None,
    L=2,
    T=50000,
    burn_in=300,
    seed=1,
    transition_type="linear",
    mixing_type="linear",
    B_density=0.6,
    B_scale=0.3,
    stable=True,
    eps_dist="gaussian",
    eps_params={"sigma": np.array([0.1] * 6)},
)

PRESET_CONFIGS = {
    "linear_gaussian": LINEAR_GAUSSIAN,
    "linear_uniform": LINEAR_UNIFORM,
    "linear_lapalce": LINEAR_LAPLACE,

    "nonlinear_transition": NONLINEAR_TRANSITION, # no use
    "three_layer_two_var": THREE_LAYER_TWO_VAR, # no use
}