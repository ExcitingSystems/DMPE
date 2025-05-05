from functools import partial
import json
import argparse
import datetime
from tqdm import tqdm

import jax
import jax.numpy as jnp

from dmpe.algorithms.algorithm_utils import interact_and_observe
from eval_dmpe import setup_env


@partial(jax.jit, static_argnums=(0, 1))
def choose_action(env, penalty_function, proposed_actions, state, choice_key):
    """Choose randomly among the proposed actions that keep the system within bounds for the next step.
    If none of the inputs keep the systems in bounds, apply the one that causes the least penalty.

    This is a heuristic implementation that uses an oracle to ensure compliance with the bounds, but chooses
    mostly randomly among the actions.
    """
    test_obs, _ = jax.vmap(env.step, in_axes=(None, 0, None))(state, proposed_actions, env.env_properties)

    penalty_values = jax.vmap(penalty_function, in_axes=(0, 0))(test_obs, proposed_actions)

    def true_fun(key, data_array, penalty_values):
        """There are no options that keep the system within bounds. Apply the one with the least penalty."""
        idx_min_penalty = jnp.argmin(penalty_values)
        return data_array[idx_min_penalty]

    def false_fun(key, data_array, penalty_values):
        """There are actions that keep the system within bounds. Choose one randomly."""
        valid_points_bool = penalty_values == 0
        prob_points = valid_points_bool.astype(jnp.float32) / jnp.sum(valid_points_bool)
        return jax.random.choice(key, data_array, p=prob_points, axis=0)

    return jax.lax.cond(
        jnp.all(penalty_values != 0), true_fun, false_fun, *(choice_key, proposed_actions, penalty_values)
    )


def run_experiment(rpm, seed):

    print(
        "Running experiment with random walk.",
        f"(seed: {int(seed)}) on the PMSM with {rpm} rpm.",
    )

    n_time_steps = 15_000
    n_tries = 5_000

    env, penalty_function = setup_env(rpm)
    obs, state = env.reset(env.env_properties)
    dim_obs_space = obs.shape[0]
    dim_action_space = env.action_dim

    observations = jnp.zeros((n_time_steps, dim_obs_space))
    observations = observations.at[0].set(obs)
    actions = jnp.zeros((n_time_steps - 1, dim_action_space))

    key = jax.random.key(seed)
    key, action_key = jax.random.split(key)

    action = jax.random.normal(action_key, shape=(env.action_dim,))

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

    # experiment finished, save results
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    with open(f"./results/heuristics/random_walk/data_rpm_{rpm}_{file_name}.json", "w") as fp:
        json.dump(dict(observations=observations.tolist(), actions=actions.tolist()), fp)

    jax.clear_caches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random walk on the PMSM environment.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use.")
    args = parser.parse_args()

    gpus = jax.devices()
    jax.config.update("jax_default_device", gpus[args.gpu_id])

    rpms = [0, 3000, 5000, 7000, 9000]

    for rpm in rpms:
        seeds = list(range(50, 61))
        for seed in seeds:
            run_experiment(rpm, seed)
