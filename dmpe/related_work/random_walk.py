from functools import partial
from tqdm import tqdm

import jax
import jax.numpy as jnp

from dmpe.algorithms.algorithm_utils import interact_and_observe


@partial(jax.jit, static_argnums=(0, 1))
def choose_action(env, penalty_function, proposed_actions, state, choice_key):
    """Choose randomly among the proposed actions that keep the system within bounds for the next step.
    If none of the inputs keep the systems in bounds, apply the one that causes the least penalty.

    This is a heursitic implmentation that uses an oracle to ensure compliance with the bounds, but chooses
    mostly randomly among the actions.
    """
    test_obs, test_state = jax.vmap(env.step, in_axes=(None, 0, None))(state, proposed_actions, env.env_properties)
    penalty_values = jax.vmap(penalty_function, in_axes=(0, 0))(test_obs[:, None, :], proposed_actions[:, None, :])

    def true_fun(key, data_array, penalty_values):
        """There are not options that keep the system within bounds. Apply the one with the least penalty."""
        idx_min_penalty = jnp.argmin(penalty_values)
        return data_array[idx_min_penalty]

    def false_fun(key, data_array, penalty_values):
        """There are actions that keep the system within bounds. Choose one randomly."""
        valid_points_bool = penalty_values == 0
        prob_points = valid_points_bool.astype(jnp.float32) / jnp.sum(valid_points_bool)
        return jax.random.choice(choice_key, proposed_actions, p=prob_points, axis=0)

    return jax.lax.cond(
        jnp.all(penalty_values != 0), true_fun, false_fun, *(choice_key, proposed_actions, penalty_values)
    )


def excite_with_random_walk(env, exp_params, key):

    n_time_steps = exp_params["n_time_steps"]
    penalty_function = exp_params["alg_params"]["penalty_function"]
    n_tries = exp_params["alg_params"]["n_tries"]

    obs, state = env.reset(env.env_properties)
    dim_obs_space = obs.shape[0]
    dim_action_space = env.action_dim

    observations = jnp.zeros((n_time_steps, dim_obs_space))
    observations = observations.at[0].set(obs)
    actions = jnp.zeros((n_time_steps - 1, dim_action_space))

    key, action_key = jax.random.split(key)
    action = jax.random.normal(action_key, shape=(dim_action_space,))

    for k in tqdm(range(n_time_steps)):

        key, action_key, choice_key = jax.random.split(key, 3)
        proposed_actions = action + jax.random.normal(
            action_key,
            shape=(
                n_tries,
                env.action_dim,
            ),
        )

        action = choose_action(env, penalty_function, proposed_actions, state, choice_key)

        next_obs, next_state, actions, observations = interact_and_observe(
            env=env, k=jnp.array([k]), action=action, state=state, actions=actions, observations=observations
        )

        state = next_state
        obs = next_obs

    return observations, actions
