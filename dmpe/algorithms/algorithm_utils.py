from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
from haiku import PRNGSequence

import exciting_environments as excenvs
from dmpe.utils.signals import aprbs
from dmpe.utils.density_estimation import select_bandwidth, DensityEstimate
from dmpe.models import NeuralEulerODE
from dmpe.excitation.excitation_utils import soft_penalty, Exciter


def consult_exciter(
    k: int,
    exciter: Exciter,
    obs: jax.Array,
    state: excenvs.CoreEnvironment.State,
    model: eqx.Module,
    density_estimate: DensityEstimate,
    proposed_actions: jax.Array,
    expl_key: jax.random.PRNGKey,
):
    """Use the exciter to choose the next action or apply the proposed actions depending on
    the current time step.

    Here, this is essentially a wrapper around the exciter's choose_action method that enables to
    overwrite the function for certain cases. Especially, this is used here to allow for random
    actions at the start of the experiment where the model is not yet trained and random actions
    are more effective compared to actions optimized based on the poor model prediction.

    Args:
        k: The current time step.
        exciter: The exciter object.
        obs: The current observation.
        state: The current state of the environment.
        model: The model used for predictions.
        density_estimate: The density estimate object.
        proposed_actions: The proposed actions for the current time step.
        expl_key: The random key for drawing new proposed actions.

    Returns:
        action: The chosen action.
        next_proposed_actions: The proposed actions for the next time step.
        next_density_estimate: The updated density estimate.
        prediction_loss: The prediction loss for the current time step.
        next_expl_key: The updated random key for drawing new proposed actions.
    """
    if k > exciter.start_optimizing:
        action, next_proposed_actions, next_density_estimate, prediction_loss, next_expl_key = exciter.choose_action(
            obs=obs,
            state=state,
            model=model,
            density_estimate=density_estimate,
            proposed_actions=proposed_actions,
            expl_key=expl_key,
        )
    else:
        # run the exciter without optimizing actions
        next_expl_key, expl_action_key = jax.random.split(expl_key, 2)
        action, next_proposed_actions, next_density_estimate = exciter.process_propositions(
            obs=obs,
            density_estimate=density_estimate,
            proposed_actions=proposed_actions,
            expl_action_key=expl_action_key,
        )
        prediction_loss = 0.0  # no prediction loss since no optimization has been performed

    return action, next_proposed_actions, next_density_estimate, prediction_loss, next_expl_key


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

    obs, state = env.step(state, action, env.env_properties)

    actions = actions.at[k].set(action)  # store u_k
    observations = observations.at[k + 1].set(obs)  # store x_{k+1}

    return obs, state, actions, observations


def default_dmpe_parameterization(
    env: excenvs.CoreEnvironment, seed: int = 0, n_time_steps=5_000, featurize=None, model_class=None
):
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
        n_prediction_steps=10,
        points_per_dim=20,
        action_lr=1e-1,
        n_opt_steps=5,
        consider_action_distribution=True,
        target_distribution=None,
        rho_obs=1,
        rho_act=1,
        penalty_order=2,
        clip_action=False,
        n_starts=20,
        reuse_proposed_actions=True,
        penalty_function=lambda x, u: soft_penalty(a=x, a_max=1, penalty_order=2)
        + soft_penalty(a=u, a_max=1, penalty_order=2),
    )

    dim = env.physical_state_dim + env.action_dim

    alg_params["target_distribution"] = jnp.ones(shape=(alg_params["points_per_dim"] ** dim, 1)) * 1 / (1 - (-1)) ** dim

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
        n_train_steps=1,
        sequence_length=alg_params["n_prediction_steps"],
        featurize=(lambda x: x) if featurize is None else featurize,
        model_lr=1e-4,
    )

    model_params = dict(
        obs_dim=env.reset(env.env_properties)[0].shape[0], action_dim=env.action_dim, width_size=128, depth=3, key=None
    )

    exp_params = dict(
        seed=seed,
        n_time_steps=n_time_steps,
        model_class=NeuralEulerODE if model_class is None else model_class,
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
