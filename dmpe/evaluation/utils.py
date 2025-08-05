import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx

from dmpe.utils.density_estimation import build_grid


def get_valid_points(data_grid, constr_func):
    valid_grid_point = jax.vmap(constr_func, in_axes=0)(data_grid) == 0
    constraint_data_points = data_grid[jnp.where(valid_grid_point == True)]
    return constraint_data_points


def valid_space_grid(constraint_function, data_dim, points_per_dim, min, max):
    hypercube_grid = build_grid(data_dim, min, max, points_per_dim)
    return get_valid_points(hypercube_grid, constraint_function)


def default_constraint_function(data_point, max_value=1, penalty_order=2):
    penalties = jax.nn.relu(jnp.abs(data_point) - max_value)
    penalties = penalties**penalty_order

    penalty = jnp.sum(penalties)
    return jnp.squeeze(penalty)
