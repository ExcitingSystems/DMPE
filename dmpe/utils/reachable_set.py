"""Utils to compute the reachable set for arbitrary non-linear systems through data driven approximation."""

from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import exciting_environments as excenvs
from dmpe.models.model_utils import simulate_ahead_with_env
from dmpe.utils.density_estimation import build_grid


def loss_function(
    actions: jax.Array,
    target: jax.Array,
    init_obs: jax.Array,
    penalty_function: Callable,
    featurize: Callable,
    env: excenvs.CoreEnvironment,
):
    init_state = env.generate_state_from_observation(init_obs, env.env_properties)
    observations, _ = simulate_ahead_with_env(env, init_obs, init_state, actions)
    # loss = jnp.min(jnp.linalg.norm(featurize(observations) - featurize(target[None]), axis=-1))
    loss = jnp.linalg.norm(featurize(observations[-1, :]) - featurize(target))
    penalties = penalty_function(observations, actions) * 1e3
    return loss + penalties


gradient_function = eqx.filter_grad(loss_function)


@eqx.filter_jit
def optimize_actions(
    proposed_actions: jax.Array,
    target: jax.Array,
    init_obs: jax.Array,
    penalty_function: Callable,
    featurize: Callable,
    env: excenvs.CoreEnvironment,
    optimizer: optax.GradientTransformation,
    n_opt_steps: int,
):
    opt_state = optimizer.init(proposed_actions)

    def body_fun(i, carry):
        proposed_actions, opt_state = carry
        grad = gradient_function(
            proposed_actions,
            target,
            init_obs,
            penalty_function,
            featurize,
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
    target: jax.Array,
    init_obs: jax.Array,
    penalty_function: Callable,
    featurize: Callable,
    env: excenvs.CoreEnvironment,
    optimizer: optax.GradientTransformation,
    n_opt_steps: int,
) -> tuple[jax.Array, jax.Array]:
    actions = eqx.filter_vmap(optimize_actions, in_axes=(0, None, None, None, None, None, None, None))(
        proposed_actions, target, init_obs, penalty_function, featurize, env, optimizer, n_opt_steps
    )

    losses = eqx.filter_vmap(loss_function, in_axes=(0, None, None, None, None, None))(
        actions, target, init_obs, penalty_function, featurize, env
    )
    best_idx = jnp.argmin(losses)
    return actions[best_idx], losses[best_idx]


def approximate_reachable_set(
    env: excenvs.CoreEnvironment,
    penalty_function: Callable,
    featurize: Callable,
    key: jax.random.PRNGKey,
    points_per_dim: int,
    sequence_length: int,
    n_starts: int,
    n_opt_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    init_obs, _ = env.reset(env.env_properties)

    obs_dim = init_obs.shape[-1]
    target_observations = build_grid(obs_dim, -1.1, 1.1, points_per_dim)

    lr = optax.schedules.exponential_decay(
        init_value=1e-1,
        transition_steps=n_opt_steps,
        transition_begin=0,
        decay_rate=0.1,
        end_value=1e-3,
    )
    optimizer = optax.adam(lr)
    proposed_actions = jax.random.uniform(
        key=key,
        shape=(target_observations.shape[0], n_starts, sequence_length, env.action_dim),
        minval=-1,
        maxval=1,
    )

    chosen_actions, losses = eqx.filter_vmap(
        optimize_actions_multistart, in_axes=(0, 0, None, None, None, None, None, None)
    )(proposed_actions, target_observations, init_obs, penalty_function, featurize, env, optimizer, n_opt_steps)

    loss_map = losses.reshape((points_per_dim, points_per_dim))
    return chosen_actions, loss_map, target_observations, proposed_actions
