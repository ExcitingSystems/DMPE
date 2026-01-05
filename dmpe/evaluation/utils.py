from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx

from dmpe.utils.density_estimation import build_grid


def get_valid_points(data_grid, constr_func):
    valid_grid_point = jax.vmap(constr_func, in_axes=0)(data_grid) == 0
    constraint_data_points = data_grid[jnp.where(valid_grid_point == True)]
    return constraint_data_points


def valid_space_grid(constraint_function, data_dim, points_per_dim, min_value, max_value):
    hypercube_grid = build_grid(data_dim, min_value, max_value, points_per_dim)
    if hypercube_grid.shape[0] > 50_000:
        out = []
        n_grid_elements = hypercube_grid.shape[0]
        chunk_size = 10_000
        for i in tqdm(jnp.arange(0, n_grid_elements, chunk_size)):
            out.append(
                get_valid_points(
                    hypercube_grid[i : min(i + chunk_size, n_grid_elements)],
                    constraint_function,
                )
            )
        return jnp.concatenate(out)
    else:
        return get_valid_points(hypercube_grid, constraint_function)


def default_constraint_function(data_point, max_value=1, penalty_order=2):
    penalties = jax.nn.relu(jnp.abs(data_point) - max_value)
    penalties = penalties**penalty_order

    penalty = jnp.sum(penalties)
    return jnp.squeeze(penalty)
