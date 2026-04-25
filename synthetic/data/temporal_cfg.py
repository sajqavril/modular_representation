import numpy as np

try:
    from synthetic.data.temporal_generator import TemporalProcessConfig
except ModuleNotFoundError:
    from temporal_generator import TemporalProcessConfig

LINEAR_GAUSSIAN = TemporalProcessConfig(
    n_layers=2,
    n_vars_per_layer=1,
    d=None,  # defaults to n
    L=1,
    T=10000,
    burn_in=2,
    seed=0,
    transition_type="linear",
    mixing_type="linear",
    B_density=1.0,
    B_scale=1.0,
    stable=False,
    eps_dist="gaussian",
    eps_params={"sigma": 0.1},
)

LINEAR_UNIFORM = TemporalProcessConfig(
    n_layers=2,
    n_vars_per_layer=1,
    d=None,  # defaults to n
    L=1,
    T=10000,
    burn_in=2,
    seed=0,
    transition_type="linear",
    mixing_type="linear",
    B_density=1.0,
    B_scale=1.0,
    stable=False,
    eps_dist="uniform",
    eps_params={"a": 0.1},
)

LINEAR_LAPLACE = TemporalProcessConfig(
    n_layers=2,
    n_vars_per_layer=1,
    d=None,  # defaults to n
    L=1,
    T=10000,
    burn_in=2,
    seed=0,
    transition_type="linear",
    mixing_type="linear",
    B_density=1.0,
    B_scale=1.0,
    stable=False,
    eps_dist="laplace",
    eps_params={"b": 0.1},
)

LG_NONLINEAR_MIXING = TemporalProcessConfig(
    n_layers=2,
    n_vars_per_layer=1,
    d=20,
    L=1,
    T=10000,
    burn_in=2,
    seed=0,
    transition_type="linear",
    mixing_type="nonlinear",
    B_density=1.0,
    B_scale=1.0,
    stable=False,
    eps_dist="gaussian",
    eps_params={"sigma": 0.1},
)

LU_NONLINEAR_MIXING = TemporalProcessConfig(
    n_layers=2,
    n_vars_per_layer=1,
    d=20,
    L=1,
    T=10000,
    burn_in=2,
    seed=0,
    transition_type="linear",
    mixing_type="nonlinear",
    B_density=1.0,
    B_scale=1.0,
    stable=False,
    eps_dist="uniform",
    eps_params={"a": 0.1},
)

LL_NONLINEAR_MIXING = TemporalProcessConfig(
    n_layers=2,
    n_vars_per_layer=1,
    d=20,
    L=1,
    T=10000,
    burn_in=2,
    seed=0,
    transition_type="linear",
    mixing_type="nonlinear",
    B_density=1.0,
    B_scale=1.0,
    stable=False,
    eps_dist="laplace",
    eps_params={"b": 0.1},
)

G_NONLINEAR = TemporalProcessConfig(
    n_layers=2,
    n_vars_per_layer=1,
    d=20,
    L=1,
    T=10000,
    burn_in=2,
    seed=0,
    transition_type="nonlinear",
    mixing_type="nonlinear",
    B_density=1.0,
    B_scale=1.0,
    stable=False,
    eps_dist="gaussian",
    eps_params={"sigma": 0.1},
)

U_NONLINEAR = TemporalProcessConfig(
    n_layers=2,
    n_vars_per_layer=1,
    d=20,
    L=1,
    T=10000,
    burn_in=2,
    seed=0,
    transition_type="nonlinear",
    mixing_type="nonlinear",
    B_density=1.0,
    B_scale=1.0,
    stable=False,
    eps_dist="uniform",
    eps_params={"a": 0.1},
)

L_NONLINEAR = TemporalProcessConfig(
    n_layers=2,
    n_vars_per_layer=1,
    d=20,
    L=1,
    T=10000,
    burn_in=2,
    seed=0,
    transition_type="nonlinear",
    mixing_type="nonlinear",
    B_density=1.0,
    B_scale=1.0,
    stable=False,
    eps_dist="laplace",
    eps_params={"b": 0.1},
)

THREE_LAYER_TWO_VAR = TemporalProcessConfig(
    n_layers=3,
    n_vars_per_layer=2,
    d=None,
    L=2,
    T=50000,
    burn_in=2,
    seed=1,
    transition_type="linear",
    mixing_type="linear",
    B_density=0.6,
    B_scale=0.3,
    stable=False,
    eps_dist="gaussian",
    eps_params={"sigma": np.array([0.1] * 6)},
)

PRESET_CONFIGS = {
    "linear_gaussian": LINEAR_GAUSSIAN,
    "linear_uniform": LINEAR_UNIFORM,
    "linear_laplace": LINEAR_LAPLACE,

    "lg_nonlinear_mixing": LG_NONLINEAR_MIXING, 
    "ll_nonlinear_mixing": LL_NONLINEAR_MIXING, 
    "lu_nonlinear_mixing": LU_NONLINEAR_MIXING, 

    "g_nonlinear": G_NONLINEAR, 
    "l_nonlinear": L_NONLINEAR, 
    "u_nonlinear": U_NONLINEAR, 

    "three_layer_two_var": THREE_LAYER_TWO_VAR, # no use
}
