from typing import Tuple, Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from haiku import PRNGSequence

import exciting_environments as excenvs
from dmpe.utils.signals import aprbs
from dmpe.utils.density_estimation import DensityEstimate, get_uniform_target_distribution
from dmpe.models.models import NeuralEulerODE
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
    """Use the exciter to choose the next action or simply apply the proposed actions depending on
    the current time step.

    This also updates the density estimate based on the chosen action. The momentary observation
    y_k has no corresponding action u_k, before the exciter has chosen one. Therefore, the
    DensityEstimate that goes into the function is still based on y_{k-1} and u_{k-1}. The
    one that is returned from this function incorporates y_k and u_k.

    This is essentially a wrapper around the exciter's choose_action method that enables
    overwriting the function for certain cases. Especially, this is used here to allow for random
    actions at the start of the experiment where the model is not yet trained and random actions
    are more effective compared to actions optimized based on the poor model prediction.

    Args:
        k (int): The current time step.
        exciter (Exciter): The exciter object.
        obs(jax.Array): The current observation.
        state(excenvs.CoreEnvironment.State): The current state of the environment.
        model (eqx.Module): The model used for predictions.
        density_estimate (DensityEstimate): The density estimate object from time step k-1.
        proposed_actions (jax.Array): The proposed actions for the current time step.
        expl_key (jax.random.PRNGKey): The random key for drawing new proposed actions.

    Returns:
        action (jax.Array): The chosen action.
        next_proposed_actions (jax.Array): The proposed actions for the next time step.
        next_density_estimate (DensityEstimate): The updated density estimate with (y_k, u_k).
        prediction_loss (float): The prediction loss for the current time step.
        next_expl_key (jax.random.PRNGKey): The updated random key for drawing new proposed actions.
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
    k: jax.Array,
    action: jax.Array,
    state: excenvs.CoreEnvironment.State,
    actions: jax.Array,
    observations: jax.Array,
) -> Tuple[jax.Array, excenvs.CoreEnvironment.State, jax.Array, jax.Array]:
    """
    Interact with the environment and store the action and the resulting observation.

    Args:
        env (excenvs.CoreEnvironment): The environment object.
        k (int): The current time step.
        action (jax.Array): The action to be taken at time step k.
        state (excenvs.CoreEnvironment.State): The state of the environment at time step k.
        actions (jax.Array): The array of actions taken so far.
        observations (jax.Array): The array of observations observed so far.

    Returns:
        obs (jax.Array): The updated observation at time step k+1.
        state (excenvs.CoreEnvironment.State): The updated state of the environment at time step k+1.
        actions (jax.Array): The updated array of actions taken so far.
        observations (jax.Array): The updated array of observations observed so far.
    """

    # apply u_k and go to x_{k+1} and observe y_{k+1}
    obs, state = env.step(state, action, env.env_properties)

    actions = actions.at[k].set(action)  # store u_k
    observations = observations.at[k + 1].set(obs)  # store y_{k+1}

    return obs, state, actions, observations


def default_dmpe_parameterization(
    env: excenvs.CoreEnvironment,
    seed: int = 0,
    n_time_steps=5_000,
    featurize: Callable | None = None,
    model_class: eqx.Module | None = None,
):
    """Returns a default parameterization for the DMPE algorithm.

    This parameterization is intended as a starting point to apply to a given system.
    The parameters are not necessarily optimal for any given system but should give a
    reasonable first impression. Currently, featurization of the model state e.g. angles
    needs to be provided manually.

    Args:
        env (excenvs.CoreEnvironment): The environment object representing the system.
        seed (int): The seed for the random number generator.
        n_time_steps (int): The number of time steps for the experiment.
        featurize (callable | None): A function to featurize the model state. Defaults to the identity function.
        model_class (eqx.Module | None): The model class to be used. Defaults to NeuralEulerODE. It must comply
            with the NeuralEulerODE API to be usable here. Otherwise, the model can also be overwritten after
            getting the other default parameters.

    Returns:
        Tuple[Dict, jax.Array, jax.random.PRNGKey, jax.random.PRNGKey]: A tuple containing the experiment parameters,
        the initial proposed actions, the key for loading data in model learning, and the key for random action generation.

    """
    alg_params = dict(
        bandwidth=0.08,
        n_prediction_steps=10,
        points_per_dim=21,
        grid_extend=1.05,
        excitation_optimizer=optax.adabelief(1e-2),
        n_opt_steps=50,
        start_optimizing=5,
        consider_action_distribution=True,
        penalty_function=None,
        target_distribution=None,
        clip_action=False,
        n_starts=10,
        reuse_proposed_actions=True,
    )

    dim = env.physical_state_dim + env.action_dim

    alg_params["penalty_function"] = lambda x, u: soft_penalty(a=x, a_max=1, penalty_order=2) + soft_penalty(
        a=u, a_max=1, penalty_order=2
    )
    alg_params["target_distribution"] = get_uniform_target_distribution(
        dim=3 if alg_params["consider_action_distribution"] else 2,
        points_per_dim=alg_params["points_per_dim"],
        bandwidth=alg_params["bandwidth"],
        grid_extend=alg_params["grid_extend"],
        consider_action_distribution=alg_params["consider_action_distribution"],
        penalty_function=alg_params["penalty_function"],
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
