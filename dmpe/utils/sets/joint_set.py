import json

import jax
import jax.numpy as jnp
import equinox as eqx

from dmpe.excitation.excitation_utils import soft_penalty


def load_results(filename: str) -> dict[str, jax.Array]:
    with open(filename, "r") as f:
        data = json.load(f)
    data = {key: jnp.array(entry) for key, entry in data.items()}
    return data


@eqx.filter_jit
def check_in_set(
    obs: jax.Array,
    set_bool: jax.Array,
    set_grid: jax.Array,
) -> jax.Array:
    """Check if the given observation is within the set.

    Args:
        obs (jax.Array): The observation to be tested with shape (obs_dim,)
        set_bool (jax.Array): The boolean array describing which grid points belong to the set with shape (points_per_dim**dim,)
        set_grid (jax.Array): The float array describing the positions of the grid points with shape (points_per_dim**dim, obs_dim)
        penalty_function (Callable): Penalty function for the observation constraints

    Returns:
        A boolean jax.Array indicating if the input belongs to the set.
    """
    dist = jnp.linalg.norm(obs[None] - set_grid, axis=-1)
    min_idx = jnp.argmin(dist)

    penalty_value = soft_penalty(obs[None])
    penalty_bool = jnp.isclose(penalty_value, 0)

    return jnp.logical_and(set_bool[min_idx], penalty_bool)
