from functools import partial

from tqdm.notebook import tqdm

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

import exciting_environments as excenvs

from model_utils import simulate_ahead, simulate_ahead_with_env
from density_estimation import update_kde_grid_multiple_observations
from metrics import JSDLoss


@jax.jit
def soft_penalty(a, a_max=1):
    """Computes penalty for the given input. Assumes symmetry in all dimensions."""
    penalty = jnp.sum(jax.nn.relu(jnp.abs(a) - a_max), axis=(-2, -1))
    return jnp.squeeze(penalty)


@eqx.filter_jit
def loss_function(
        model,
        init_obs: jnp.ndarray,
        init_state: jnp.ndarray,
        actions: jnp.ndarray,
        p_est: jnp.ndarray,
        x: jnp.ndarray,
        start_n_measurments: jnp.ndarray,
        bandwidth: float,
        tau: float,
        target_distribution: jnp.ndarray
    ) -> jnp.ndarray:
    """Predicts a trajectory based on the given actions and the model and computes the
    corresponding loss value.

    Args:
        model: The model to use for the prediction
        init_obs: The initial observation from which to start the simulation
        init_state: The initial state from which to start the simulation
        actions: The actions to apply in each step of the simulation, the length
            of the first dimension of this array determine the lenght of the
            output.
        p_est: The current estimation for the probability density
        x: The grid coordinates for the density estimation
        start_n_measurements: The number of measurements actually gathered from the
            environment so far
        bandwidth: The bandwidth of the density estimation
        tau: The sampling time for the model
        target_distribution: The goal distribution of the data. The JSD loss is computed
            w.r.t. this distribution
    """

    if isinstance(model, excenvs.core_env.CoreEnvironment):
        observations = simulate_ahead_with_env(
            env=model,
            init_obs=init_obs,
            init_state=init_state,
            actions=actions,
            env_state_normalizer=model.env_state_normalizer[0, :],
            action_normalizer=model.action_normalizer[0, :],
            static_params={key: value[0, :] for (key, value) in model.static_params.items()}
        )
    else:
        observations = simulate_ahead(
            model=model,
            init_obs=init_obs,
            actions=actions,
            tau=tau
        )

    p_est = update_kde_grid_multiple_observations(p_est, x, observations, start_n_measurments, bandwidth)    
    loss = JSDLoss(
        p=p_est,
        q=target_distribution
    )

    # TODO: pull this automatically, maybe penalty_kwargs or something
    rho_obs = 1e4
    rho_act = 1e4
    penalty_terms = rho_obs * soft_penalty(a=observations, a_max=1) + rho_act * soft_penalty(a=actions, a_max=1)

    return loss + penalty_terms


def optimize(
        grad_loss_function,
        proposed_actions,
        model,
        init_obs,
        init_state,
        p_est,
        x,
        start_n_measurments,
        bandwidth,
        tau,
        target_distribution
    ):

    solver = optax.adabelief(learning_rate=1e-1)
    opt_state = solver.init(proposed_actions)

    for iter in range(5):
        grad = jax.vmap(
            grad_loss_function,
            in_axes=(None, 0, 0, 0, 0, None, None, None, None, None)
        )(
            model,
            init_obs,
            init_state,
            proposed_actions,
            p_est,
            x,
            start_n_measurments,
            bandwidth,
            tau,
            target_distribution
        )
        updates, opt_state = solver.update(grad, opt_state, proposed_actions)
        proposed_actions = optax.apply_updates(proposed_actions, updates)

    final_loss = jax.vmap(
            loss_function,
            in_axes=(None, 0, 0, 0, 0, None, None, None, None, None)
        )(
            model,
            init_obs,
            init_state,
            proposed_actions,
            p_est,
            x,
            start_n_measurments,
            bandwidth,
            tau,
            target_distribution
        )

    return proposed_actions, final_loss
