from functools import partial
import json
import time
import argparse
import datetime
import pathlib

import numpy as np
import jax
import jax.numpy as jnp

from dmpe.utils.density_estimation import build_grid
from dmpe.utils.env_utils.foc_pi import ClassicController
from dmpe.evaluation.scripts.pmsm.eval_dmpe import setup_env, TARGETED_DATA_PATH


@partial(jax.jit, static_argnums=(0, 1))
def run_pi_experiment(env, pi, references_norm, init_obs, init_state, init_pi_state):
    """Rollout the experiment by following the given references with the PI controller."""

    def body_fun(carry, reference_norm):
        obs, state, pi_state = carry

        currents_norm = obs[None]
        eps = state.physical_state.epsilon * jnp.ones((1, 1))
        reference_norm = reference_norm[None, ...]
        pi_obs = jnp.concatenate([currents_norm, eps, reference_norm], axis=-1)

        action, next_pi_state = pi(pi_obs, pi_state)

        next_obs, next_state = env.step(state, jnp.squeeze(action), env.env_properties)
        return (next_obs, next_state, next_pi_state), jnp.array([jnp.squeeze(obs), jnp.squeeze(action)])

    (_, _, _), data = jax.lax.scan(body_fun, (init_obs, init_state, init_pi_state), references_norm)
    observations = data[:, 0, :]
    actions = data[:, 1, :]

    return observations, actions


def subsample_references(references, target_N):
    M = references.shape[0]

    x_old = jnp.linspace(0, M - 1, M)
    x_new = jnp.linspace(0, M - 1, target_N)

    def linear_interp(x, xp, fp):
        idx = jnp.searchsorted(xp, x) - 1
        idx = jnp.clip(idx, 0, len(fp) - 2)
        x0, x1 = xp[idx], xp[idx + 1]
        y0, y1 = fp[idx], fp[idx + 1]
        return y0 + (y1 - y0) * (x - x0)[:, None] / (x1 - x0)[:, None]

    return linear_interp(x_new, x_old, references)


def induced_voltage_constr(z_g, env, w):
    """Compute voltage constraint violations."""
    r_s = env.env_properties.static_params.r_s

    i_d_normalizer = env.env_properties.physical_normalizations.i_d
    i_q_normalizer = env.env_properties.physical_normalizations.i_q
    physical_i_d = i_d_normalizer.denormalize(z_g[0])
    physical_i_q = i_q_normalizer.denormalize(z_g[1])

    psid = env.LUT_interpolators["Psi_d"](jnp.array([physical_i_d, physical_i_q]))
    psiq = env.LUT_interpolators["Psi_q"](jnp.array([physical_i_d, physical_i_q]))

    ud = r_s * physical_i_d - w * psiq
    uq = r_s * physical_i_q + w * psid

    u = jnp.sqrt(ud**2 + uq**2)

    return jnp.squeeze(jax.nn.relu(u - 400 / jnp.sqrt(3)))


def filter_voltage_constraints(env, rpm, references):
    """Filter out references that violate the voltage constraints."""
    penalty_values = jax.vmap(induced_voltage_constr, in_axes=(0, None, None))(
        references, env, rpm * env.env_properties.static_params.p * 2 * jnp.pi / 60
    )

    valid_points = penalty_values == 0
    filtered_references = references[jnp.where(valid_points == True)]

    return filtered_references


def setup_references(env, rpm, points_per_dim, penalty_function, n_timesteps=15_000):
    """Setup a reference trajectory to follow with the PI controller."""
    references = build_grid(2, low=-0.95, high=0.95, points_per_dim=points_per_dim)
    references = jnp.flip(references, axis=1)
    references = jnp.flip(references, axis=0)

    references = references.reshape(points_per_dim, points_per_dim, 2)
    references = np.array(references)
    for k in range(points_per_dim):
        if k % 2 == 0:
            references[k] = np.flip(references[k], axis=0)
    references = references.reshape(points_per_dim**2, 2)
    references = references[int(points_per_dim / 2) :]

    references = jnp.array(references)

    references_norm = references[:, None, :].repeat(5, axis=1).reshape(-1, 2)
    # references_norm = references[:, None, :].repeat(500, axis=1).reshape(-1, 2)

    def _filter_valid_points(data_points, penalty_function):

        valid_points_bool = jax.vmap(penalty_function, in_axes=0)(data_points) == 0
        return data_points[jnp.where(valid_points_bool == True)]

    filter_valid_points = lambda observations: _filter_valid_points(
        observations, penalty_function=lambda x: penalty_function(x, None)
    )

    references_norm = filter_valid_points(references_norm)
    references_norm = filter_voltage_constraints(env, rpm, references_norm)
    references_norm = subsample_references(references_norm, n_timesteps)
    return references_norm


def run_experiment(rpm):
    print(
        "Running experiment with PI controller",
        f"on the PMSM with {rpm} rpm.",
    )

    # Check that the targeted data folder actually exist:
    results_path = TARGETED_DATA_PATH / pathlib.Path("heuristics") / pathlib.Path("current_plane_sweep")
    print(f"Results will be written to: '{results_path}'.")
    assert results_path.exists(), (
        f"The expected results path '{results_path}' does not seem to exist. Please create the necessary file structure "
        + "or adapt the path."
    )

    points_per_dim = 100

    env, penalty_function = setup_env(rpm)

    pi = ClassicController(
        motor=env, rpm=rpm, saturated=env.env_properties.saturated, a=4, decoupling=True, tau=env.tau
    )

    references = setup_references(env, rpm, points_per_dim, penalty_function)

    init_obs, init_state = env.reset(env.env_properties)
    init_pi_state = pi.reset(1)

    start = time.time()
    observations, actions = run_pi_experiment(env, pi, references, init_obs, init_state, init_pi_state)
    end = time.time()
    print("computation_time:", round(end - start, 4), "s")

    # experiment finished, save results
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with open(results_path / pathlib.Path(f"data_rpm_{rpm}_{file_name}.json"), "w") as fp:
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
        # no need for a seed since the experiment is deterministic
        run_experiment(rpm)
