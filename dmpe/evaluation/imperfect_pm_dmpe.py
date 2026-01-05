from typing import Callable
from copy import deepcopy

import jax
import jax.numpy as jnp
import optax
from haiku import PRNGSequence
import jax_dataclasses as jdc

import exciting_environments as excenvs

from dmpe.utils.signals import aprbs
from dmpe.algorithms.algorithms import excite_with_dmpe
from dmpe.utils.density_estimation import select_bandwidth, get_uniform_target_distribution
from dmpe.utils.env_utils.fluid_tank_utils import setup_env as setup_fluid_tank_env
from dmpe.utils.env_utils.pendulum_utils import setup_env as setup_pendulum_env
from dmpe.utils.env_utils.cart_pole_utils import setup_env as setup_cart_pole_env


def adapt_static_params(env, new_static_params):
    adapted_env = deepcopy(env)
    with jdc.copy_and_mutate(env.env_properties, validate=True) as new_env_properties:
        new_env_properties.static_params = new_static_params
    adapted_env.env_properties = new_env_properties
    return adapted_env


def get_experiment_params(
    seed: int,
    n_time_steps: int,
    env: excenvs.CoreEnvironment,
    penalty_function: Callable,
    env_params: dict,
    static_params: excenvs.CoreEnvironment.StaticParams,
):
    if type(env) == excenvs.FluidTank:
        alg_params = dict(
            bandwidth=None,
            n_prediction_steps=10,
            points_per_dim=50,
            grid_extend=1.05,
            excitation_optimizer=optax.adabelief(1e-1),
            n_opt_steps=10,
            start_optimizing=5,
            consider_action_distribution=True,
            penalty_function=penalty_function,
            target_distribution=None,
            clip_action=True,
            n_starts=5,
            reuse_proposed_actions=True,
        )

        alg_params["bandwidth"] = float(
            select_bandwidth(
                delta_z=2,
                dim=env.physical_state_dim + env.action_dim,
                n_g=alg_params["points_per_dim"],
                percentage=0.3,
            )
        )

        # overwrite and target distribution
        alg_params["target_distribution"] = get_uniform_target_distribution(
            dim=2 if alg_params["consider_action_distribution"] else 1,
            points_per_dim=alg_params["points_per_dim"],
            bandwidth=alg_params["bandwidth"],
            grid_extend=alg_params["grid_extend"],
            consider_action_distribution=alg_params["consider_action_distribution"],
            penalty_function=alg_params["penalty_function"],
        )

        exp_params = dict(
            seed=seed,
            n_time_steps=n_time_steps,
            model_class=adapt_static_params,
            env_params=env_params,
            alg_params=alg_params,
            model_trainer_params=None,
            model_params=dict(env=env, new_static_params=static_params),
        )

    elif type(env) == excenvs.Pendulum:
        alg_params = dict(
            bandwidth=None,
            n_prediction_steps=20,
            points_per_dim=21,
            grid_extend=1.05,
            excitation_optimizer=optax.adabelief(1e-1),
            n_opt_steps=10,
            start_optimizing=5,
            consider_action_distribution=True,
            penalty_function=penalty_function,
            target_distribution=None,
            clip_action=True,
            n_starts=5,
            reuse_proposed_actions=True,
        )

        alg_params["bandwidth"] = float(
            select_bandwidth(
                delta_z=2,
                dim=env.physical_state_dim + env.action_dim,
                n_g=alg_params["points_per_dim"],
                percentage=0.3,
            )
        )

        # overwrite target distribution
        alg_params["target_distribution"] = get_uniform_target_distribution(
            dim=3 if alg_params["consider_action_distribution"] else 2,
            points_per_dim=alg_params["points_per_dim"],
            bandwidth=alg_params["bandwidth"],
            grid_extend=alg_params["grid_extend"],
            consider_action_distribution=alg_params["consider_action_distribution"],
            penalty_function=alg_params["penalty_function"],
        )

        exp_params = dict(
            seed=seed,
            n_time_steps=n_time_steps,
            model_class=adapt_static_params,
            env_params=env_params,
            alg_params=alg_params,
            model_trainer_params=None,
            model_params=dict(env=env, new_static_params=static_params),
        )
    elif type(env) == excenvs.CartPole:
        alg_params = dict(
            bandwidth=0.12,
            n_prediction_steps=50,
            points_per_dim=10,
            grid_extend=1.05,
            excitation_optimizer=optax.adabelief(1e-1),
            n_opt_steps=5,
            start_optimizing=5,
            consider_action_distribution=True,
            penalty_function=penalty_function,
            target_distribution=None,
            clip_action=True,
            n_starts=5,
            reuse_proposed_actions=True,
        )
        # overwrite target distribution
        alg_params["target_distribution"] = get_uniform_target_distribution(
            dim=5 if alg_params["consider_action_distribution"] else 4,
            points_per_dim=alg_params["points_per_dim"],
            bandwidth=alg_params["bandwidth"],
            grid_extend=alg_params["grid_extend"],
            consider_action_distribution=alg_params["consider_action_distribution"],
            penalty_function=alg_params["penalty_function"],
        )

        exp_params = dict(
            seed=seed,
            n_time_steps=n_time_steps,
            model_class=adapt_static_params,
            env_params=env_params,
            alg_params=alg_params,
            model_trainer_params=None,
            model_params=dict(env=env, new_static_params=static_params),
        )
    else:
        raise NotADirectoryError(f"There are no parameters for the given env of type {type(env)}.")

    return env, exp_params


def excite_with_imperfect_model(
    seed: int,
    n_time_steps: int,
    env: excenvs.CoreEnvironment,
    penalty_function: Callable,
    env_params: dict,
    static_params: excenvs.CoreEnvironment.StaticParams,
):
    env, exp_params = get_experiment_params(
        seed,
        n_time_steps,
        env,
        penalty_function,
        env_params,
        static_params,
    )

    alg_params = exp_params["alg_params"]

    key = jax.random.PRNGKey(seed=exp_params["seed"])
    data_key, _, _, expl_key, key = jax.random.split(key, 5)
    data_rng = PRNGSequence(data_key)

    # initial guess
    proposed_actions = jnp.hstack(
        [
            aprbs(alg_params["n_prediction_steps"], env.batch_size, 1, 10, next(data_rng))[0]
            for _ in range(env.action_dim)
        ]
    )

    observations, actions, model, density_estimate, losses, proposed_actions, callback_out = excite_with_dmpe(
        env=env,
        exp_params=exp_params,
        proposed_actions=proposed_actions,
        loader_key=None,
        expl_key=expl_key,
    )

    return observations, actions, model, exp_params, (density_estimate, losses, proposed_actions, callback_out)
