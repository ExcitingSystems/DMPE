import jax
import jax.numpy as jnp

import equinox as eqx


@jax.jit
def gaussian_kernel(x: jnp.ndarray, bandwidth: float) -> jnp.ndarray:
    """Evaluates the Gaussian RBF kernel at x with given bandwidth. This can take arbitrary
    dimensions for 'x' and will compute the output by broadcasting. The last dimension of
    the input needs to be the dimension of the data which is reduced.
    """
    data_dim = x.shape[-1]
    factor = bandwidth**data_dim * jnp.power(2 * jnp.pi, data_dim / 2)
    return 1 / factor * jnp.exp(- jnp.linalg.norm(x, axis=-1)**2 / (2*bandwidth**2))


class DensityEstimate(eqx.Module):
    """Holds an estimation of the density of sampled datapoints.
    
    Args:
        p: The probability estimates at the grid points
        x_g: The grid points
        bandwidth: The bandwidth of the kernel density estimate
        n_observations: The number of observations that make up the current
            estimate
    """

    p: jnp.float32
    x_g: jnp.ndarray
    bandwidth: jnp.ndarray
    n_observations: jnp.ndarray

    @classmethod
    def from_estimate(cls, p, n_additional_observations, density_estimate):
        return cls(
            p=p,
            n_observations=(density_estimate.n_observations + n_additional_observations),
            x_g=density_estimate.x_g,
            bandwidth=density_estimate.bandwidth
        )


@jax.jit
def update_density_estimate_single_observation(
        density_estimate: DensityEstimate,
        observation: jnp.ndarray
) -> jnp.ndarray:
    """Recursive update to the kernel density estimation (KDE) on a fixed grid.

    Args:
        density_estimate: The density estimate before the update
        observation: The new data point

    Returns:
        The updated density estimate
    """
    kernel_value = gaussian_kernel(
        x=density_estimate.x_g - observation,
        bandwidth=density_estimate.bandwidth
    )
    p_est = (1 / (density_estimate.n_observations + 1) 
             * (density_estimate.n_observations * density_estimate.p + kernel_value[..., None]))

    return DensityEstimate.from_estimate(
        p=p_est,
        n_additional_observations=1,
        density_estimate=density_estimate
    )


@jax.jit
def update_density_estimate_multiple_observations(
        density_estimate: DensityEstimate,
        observations: jnp.ndarray
) -> jnp.ndarray:
    """Add a new sequence of observations to the current data density estimate.

    Args:
        density_estimate: The density estimate before the update
        observations: The sequence of observations

    Returns:
        The updated values for the density estimate
    """

    def shifted_gaussian_kernel(x, observation, bandwidth):
        return gaussian_kernel(x - observation, bandwidth)

    new_sum_part = jax.vmap(shifted_gaussian_kernel, in_axes=(None, 0, None))(
        density_estimate.x_g, observations, density_estimate.bandwidth
    )
    new_sum_part = jnp.sum(new_sum_part, axis=0)[..., None]
    p_est = (1 / (density_estimate.n_observations + observations.shape[0])
             * (density_estimate.n_observations * density_estimate.p + new_sum_part))

    return DensityEstimate.from_estimate(
        p=p_est,
        n_additional_observations=observations.shape[0],
        density_estimate=density_estimate
    )


# TODO: implement a more general build_grid function for arbitrary dims and constraints

def build_grid_2d(low, high, points_per_dim):
    x1, x2 = [
        jnp.linspace(low, high, points_per_dim),
        jnp.linspace(low, high, points_per_dim)
    ]

    x_g = jnp.meshgrid(*[x1, x2])
    x_g = jnp.stack([_x for _x in x_g], axis=-1)
    x_g = x_g.reshape(-1, 2)

    assert x_g.shape[0] == points_per_dim**2
    return x_g


def build_grid_3d(low, high, points_per_dim):
    x1, x2, x3 = [
        jnp.linspace(low, high, points_per_dim),
        jnp.linspace(low, high, points_per_dim),
        jnp.linspace(low, high, points_per_dim)
    ]

    x_g = jnp.meshgrid(*[x1, x2, x3])
    x_g = jnp.stack([_x for _x in x_g], axis=-1)
    x_g = x_g.reshape(-1, 3)

    assert x_g.shape[0] == points_per_dim**3
    return x_g
