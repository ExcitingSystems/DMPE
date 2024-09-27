from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
from haiku import PRNGSequence

import exciting_environments as excenvs
from exciting_exciting_systems.utils.signals import aprbs
from exciting_exciting_systems.utils.density_estimation import select_bandwidth
from exciting_exciting_systems.models import NeuralEulerODE


@eqx.filter_jit
def interact_and_observe(
    env: excenvs.CoreEnvironment,
    k: int,
    action: jax.Array,
    state: excenvs.CoreEnvironment.State,
    actions: jax.Array,
    observations: jax.Array,
) -> Tuple[jax.Array, excenvs.CoreEnvironment.State, jax.Array, jax.Array]:
    """
    Interact with the environment and store the action and the resulting observation.

    Args:
        env: The environment object.
        k: The current time step.
        action: The action to be taken at time step k.
        state: The state of the environment at time step k.
        actions: The list of actions taken so far.
        observations: The list of observations observed so far.

    Returns:
        obs: The updated observation at time step k+1.
        state: The updated state of the environment at time step k+1.
        actions: The updated list of actions taken so far.
        observations: The updated list of observations observed so far.
    """

    # apply u_k and go to x_{k+1}
    obs, _, _, _, state = env.step(state, action, env.env_properties)

    actions = actions.at[k].set(action)  # store u_k
    observations = observations.at[k + 1].set(obs)  # store x_{k+1}

    return obs, state, actions, observations


def default_dmpe_parameterization(env: excenvs.CoreEnvironment, seed: int = 0):
    """Returns a default parameterization for the DMPE algorithm.

    This parameterization is intended as a starting point to apply to a given system.
    The parameters are not necessarily optimal for any given system but should give a
    reasonable first impression. Currently, featurization of the model state e.g. angles
    needs to be provided manually.

    In future work, automatic tuning for the parameters will be added such that no
    manual tuning is required.

    Args:
        env (excenvs.CoreEnvironment): The environment object representing the system.
        seed (int): The seed for the random number generator.

    Returns:
        Tuple[Dict, jax.Array, jax.random.PRNGKey, jax.random.PRNGKey]: A tuple containing the experiment parameters,
        the initial proposed actions, the key for loading data in model learning, and the key for random action generation.

    """
    alg_params = dict(
        bandwidth=None,
        n_prediction_steps=50,
        points_per_dim=50,
        action_lr=1e-1,
        n_opt_steps=10,
        rho_obs=1,
        rho_act=1,
        penalty_order=2,
        clip_action=True,
        n_starts=5,
        reuse_proposed_actions=True,
    )
    alg_params["bandwidth"] = float(
        select_bandwidth(
            delta_x=2,
            dim=env.physical_state_dim + env.action_dim,
            n_g=alg_params["points_per_dim"],
            percentage=0.3,
        )
    )

    model_trainer_params = dict(
        start_learning=alg_params["n_prediction_steps"],
        training_batch_size=128,
        n_train_steps=3,
        sequence_length=alg_params["n_prediction_steps"],
        featurize=lambda x: x,
        model_lr=1e-4,
    )
    model_params = dict(obs_dim=env.physical_state_dim, action_dim=env.action_dim, width_size=128, depth=3, key=None)

    exp_params = dict(
        seed=seed,
        n_time_steps=15_000,
        model_class=NeuralEulerODE,
        env_params=None,
        alg_params=alg_params,
        model_trainer_params=model_trainer_params,
        model_params=model_params,
    )

    key = jax.random.PRNGKey(seed=exp_params["seed"])
    data_key, model_key, loader_key, expl_key, key = jax.random.split(key, 5)

    data_rng = PRNGSequence(data_key)
    exp_params["model_params"]["key"] = model_key

    # initial guess
    proposed_actions = jnp.hstack(
        [
            aprbs(alg_params["n_prediction_steps"], env.batch_size, 1, 10, next(data_rng))[0]
            for _ in range(env.action_dim)
        ]
    )

    return exp_params, proposed_actions, loader_key, expl_key
