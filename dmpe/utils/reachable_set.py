"""Utils to compute the reachable set for arbitrary non-linear systems through data driven approximation."""

from typing import Callable
import json

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import jax_tqdm

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
    sq_distances = jnp.sum((featurize(observations) - featurize(target[None])) ** 2, axis=-1)
    loss = jnp.min(sq_distances)
    # loss = jnp.sum((featurize(observations[-1, :]) - featurize(target)) ** 2)
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

    @jax_tqdm.loop_tqdm(n_opt_steps)
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
    target_observations: jax.Array,
    penalty_function: Callable,
    featurize: Callable,
    key: jax.random.PRNGKey,
    sequence_length: int,
    n_starts: int,
    n_opt_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    init_obs, _ = env.reset(env.env_properties)

    lr = optax.schedules.exponential_decay(
        init_value=1e-1,
        transition_steps=int(n_opt_steps / 3),
        transition_begin=0,
        decay_rate=0.1,
        end_value=1e-4,
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

    return chosen_actions, losses, proposed_actions


def evaluate_reachability(obs: jax.Array, reach_set: jax.Array) -> jax.Array:
    """"""
    dist = jnp.linalg.norm(obs[None] - reach_set)

    raise NotImplementedError


def save_results(filename: str, chosen_actions, losses, target_observations):
    data = dict(
        chosen_actions_rs=chosen_actions.tolist(),
        losses_rs=losses.tolist(),
        target_observations_rs=target_observations.tolist(),
    )
    with open(filename, "w") as f:
        json.dump(data, f)


def load_results(filename: str) -> dict[str, jax.Array]:
    with open(filename, "r") as f:
        data = json.load(f)
    data = {key: jnp.array(entry) for key, entry in data.items()}
    return data
