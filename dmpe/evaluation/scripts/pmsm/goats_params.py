from typing import Callable

import jax
import jax.numpy as jnp
import optax


def get_alg_params(consider_action_distribution: bool, penalty_function: Callable):
    """Get parameters for the iGOATS algorithm in the PMSM experiments."""
    h = 4
    a = 4

    alg_params = dict(
        consider_action_distribution=consider_action_distribution,
        prediction_horizon=h,
        application_horizon=a,
        bounds_amplitude=(-1, 1),
        bounds_duration=(1, 50),
        population_size=50,
        n_generations=25,
        featurize=lambda x: x,
        rng=None,
        compress_data=True,
        compression_target_N=500,
        compression_feat_dim=0,
        compression_dist_th=0.1,
        penalty_function=penalty_function,
    )

    return alg_params
