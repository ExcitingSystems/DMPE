import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx


def build_grid(dim, low, high, points_per_dim):
    """Build a uniform grid of points in the given dimension."""
    xs = [jnp.linspace(low, high, points_per_dim) for _ in range(dim)]

    x_g = jnp.meshgrid(*xs)
    x_g = jnp.stack([_x for _x in x_g], axis=-1)
    x_g = x_g.reshape(-1, dim)

    assert x_g.shape[0] == points_per_dim**dim
    return x_g


def get_valid_points(data_grid, constr_func):
    valid_grid_point = jax.vmap(constr_func, in_axes=0)(data_grid) == 0
    constraint_data_points = data_grid[jnp.where(valid_grid_point == True)]
    return constraint_data_points


def valid_space_grid(constraint_function, data_dim, points_per_dim, min, max):
    hypercube_grid = build_grid(data_dim, min, max, points_per_dim)
    return get_valid_points(hypercube_grid, constraint_function)
