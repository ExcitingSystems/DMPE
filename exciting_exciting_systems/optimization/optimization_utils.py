import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from exciting_environments.core_env import CoreEnvironment

from exciting_exciting_systems.models.model_utils import simulate_ahead, simulate_ahead_with_env
from exciting_exciting_systems.utils.density_estimation import (
    update_kde_grid_multiple_observations, update_kde_grid
)
from exciting_exciting_systems.utils.metrics import JSDLoss


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

    if isinstance(model, CoreEnvironment):
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


@eqx.filter_jit
def optimize(
        grad_loss_function,
        proposed_actions,
        model,
        solver,
        init_obs,
        init_state,
        p_est,
        x,
        start_n_measurments,
        bandwidth,
        tau,
        target_distribution
):
    opt_state = solver.init(proposed_actions)

    def body_fun(i, carry):
        proposed_actions, opt_state = carry
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
        return (proposed_actions, opt_state)

    proposed_actions, _ = jax.lax.fori_loop(0, 5, body_fun, (proposed_actions, opt_state))
    return proposed_actions


@eqx.filter_jit
def choose_action(
        grad_loss_function,
        proposed_actions,
        model,
        solver_prediction,
        init_obs,
        init_state,
        p_est,
        x_g,
        start_n_measurments,
        bandwidth,
        tau,
        target_distribution
):
    """Chooses which action to apply and updated the underlying density estimate."""

    proposed_actions = optimize(
        grad_loss_function=grad_loss_function,
        proposed_actions=proposed_actions,
        model=model,
        solver=solver_prediction,
        init_obs=init_obs,
        init_state=init_state,
        p_est=p_est,
        x=x_g,
        start_n_measurments=start_n_measurments,
        bandwidth=bandwidth,
        tau=tau,
        target_distribution=target_distribution
    )

    # update grid KDE with x_k
    p_est = jax.vmap(update_kde_grid, in_axes=[0, None, 0, None, None])(
        p_est, x_g, init_obs, start_n_measurments, bandwidth
    )

    action = proposed_actions[:, 0, :]
    proposed_actions = proposed_actions.at[:, :-1, :].set(proposed_actions[:, 1:, :])

    return action, proposed_actions, p_est
