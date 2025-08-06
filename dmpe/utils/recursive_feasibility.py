"""Utils to compute the recursive feasible set for arbitrary non-linear systems through data driven approximation."""

from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import exciting_environments as excenvs
from dmpe.models.model_utils import simulate_ahead_with_env


def loss_function(actions: jax.Array, init_obs: jax.Array, penalty_function: Callable, env: excenvs.CoreEnvironment):
    init_state = env.generate_state_from_observation(init_obs, env.env_properties)
    observations, _ = simulate_ahead_with_env(env, init_obs, init_state, actions)
    return penalty_function(observations, actions)


gradient_function = eqx.filter_grad(loss_function)


@eqx.filter_jit
def optimize_actions(
    proposed_actions: jax.Array,
    init_obs: jax.Array,
    penalty_function: Callable,
    env: excenvs.CoreEnvironment,
    optimizer: optax.GradientTransformation,
    n_opt_steps: int,
):
    opt_state = optimizer.init(proposed_actions)

    def body_fun(i, carry):
        proposed_actions, opt_state = carry
        grad = gradient_function(
            proposed_actions,
            init_obs,
            penalty_function,
            env,
        )
        updates, opt_state = optimizer.update(grad, opt_state, proposed_actions)
        proposed_actions = optax.apply_updates(proposed_actions, updates)

        return (proposed_actions, opt_state)

    proposed_actions, _ = jax.lax.fori_loop(0, n_opt_steps, body_fun, (proposed_actions, opt_state))
    return proposed_actions


@eqx.filter_jit
def optimize_actions_multistart(
    proposed_actions: jax.Array,
    init_obs: jax.Array,
    penalty_function: Callable,
    env: excenvs.CoreEnvironment,
    optimizer: optax.GradientTransformation,
    n_opt_steps: int,
):
    actions = eqx.filter_vmap(optimize_actions, in_axes=(0, None, None, None, None, None))(
        proposed_actions, init_obs, penalty_function, env, optimizer, n_opt_steps
    )

    losses = eqx.filter_vmap(loss_function, in_axes=(0, None, None, None))(actions, init_obs, penalty_function, env)
    best_idx = jnp.argmin(losses)
    return actions[best_idx], losses[best_idx]
