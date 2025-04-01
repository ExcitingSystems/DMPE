from typing import Callable

import jax
import jax.numpy as jnp
import optax

from dmpe.models.models import NeuralEulerODEPMSM
from dmpe.models.model_utils import ModelEnvWrapperPMSM
from forks.DMPE.dmpe.models.rls import SimulationPMSM_RLS
from dmpe.utils.density_estimation import build_grid, DensityEstimate


def get_target_distribution(
    points_per_dim: int,
    bandwidth: float,
    grid_extend: float,
    consider_action_distribution: bool,
    penalty_function: Callable,
):
    """Get the target distribution for the DMPE algorithm in the PMSM experiments."""

    dim = 4 if consider_action_distribution else 2
    x_g = build_grid(dim, low=-grid_extend, high=grid_extend, points_per_dim=points_per_dim)

    if consider_action_distribution:
        constr_func = lambda x_g: penalty_function(x_g[..., None, :2], x_g[..., None, 2:])
    else:
        constr_func = lambda x_g: penalty_function(x_g[..., None, :2], None)

    valid_grid_point = jax.vmap(constr_func, in_axes=0)(x_g) == 0
    constrained_data_points = x_g[jnp.where(valid_grid_point == True)]

    target_distribution = DensityEstimate.from_dataset(
        constrained_data_points[None],
        x_min=-grid_extend,
        x_max=grid_extend,
        points_per_dim=points_per_dim,
        bandwidth=bandwidth,
    )
    return target_distribution.p[0] / jnp.sum(target_distribution.p[0])


def get_alg_params(consider_action_distribution: bool, penalty_function: Callable):
    """Get parameters for the DMPE algorithm in the PMSM experiments."""

    alg_params = dict(
        bandwidth=0.08,
        n_prediction_steps=5,
        points_per_dim=21,
        grid_extend=1.05,
        excitation_optimizer=optax.adabelief(1e-2),
        n_opt_steps=200,
        start_optimizing=5,
        consider_action_distribution=consider_action_distribution,
        penalty_function=penalty_function,
        target_distribution=None,
        clip_action=False,
        n_starts=10,
        reuse_proposed_actions=True,
    )

    alg_params["target_distribution"] = get_target_distribution(
        points_per_dim=alg_params["points_per_dim"],
        bandwidth=alg_params["bandwidth"],
        grid_extend=alg_params["grid_extend"],
        consider_action_distribution=consider_action_distribution,
        penalty_function=penalty_function,
    )
    return alg_params


def get_RLS_params(consider_action_distribution, penalty_function):
    """Get parameters for the application of RLS models for the DMPE pmsm experiments."""
    alg_params = get_alg_params(
        consider_action_distribution=consider_action_distribution, penalty_function=penalty_function
    )

    model_params = dict(lambda_=0.9)
    model_trainer_params = None
    model_class = SimulationPMSM_RLS
    model_env_wrapper = None

    return alg_params, model_params, model_class, model_trainer_params, model_env_wrapper


def get_NODE_params(consider_action_distribution, penalty_function):
    """Get parameters for the application of NODE models for the DMPE pmsm experiments."""

    alg_params = get_alg_params(
        consider_action_distribution=consider_action_distribution, penalty_function=penalty_function
    )

    model_params = dict(obs_dim=2, action_dim=2, width_size=64, depth=3, key=None)
    model_trainer_params = dict(
        start_learning=alg_params["n_prediction_steps"],
        training_batch_size=64,
        n_train_steps=5,
        sequence_length=alg_params["n_prediction_steps"],
        featurize=lambda x: x,
        model_lr=1e-4,
    )
    model_class = NeuralEulerODEPMSM
    model_env_wrapper = None

    return alg_params, model_params, model_class, model_trainer_params, model_env_wrapper


def get_PM_params(consider_action_distribution, penalty_function):
    alg_params = get_alg_params(
        consider_action_distribution=consider_action_distribution, penalty_function=penalty_function
    )

    alg_params["n_opt_steps"] = 100

    model_params = None
    model_trainer_params = None
    model_class = None
    model_env_wrapper = ModelEnvWrapperPMSM

    return alg_params, model_params, model_class, model_trainer_params, model_env_wrapper
