import jax
import jax.numpy as jnp


@jax.jit
def gaussian_kernel(x: jnp.ndarray, bandwidth: float) -> jnp.ndarray:
    """Evaluates the Gaussian RBF kernel at x with given bandwidth. This can take arbitrary
    dimensions for 'x' and will compute the output by broadcasting. The last dimension of
    the input needs to be the dimension of the data which is reduced.
    """
    data_dim = x.shape[-1]
    factor = bandwidth**data_dim * jnp.power(2 * jnp.pi, data_dim / 2)
    return 1 / factor * jnp.exp(- jnp.linalg.norm(x, axis=-1)**2 / (2*bandwidth**2))


@jax.jit
def update_kde_grid(
        kde_grid: jnp.ndarray,
        x_eval: jnp.ndarray,
        measurement: jnp.ndarray,
        n_measurements: int,
        bandwidth: float,
    ) -> jnp.ndarray:
    """Recursive update to the kernel density estimation (KDE) on a fixed grid.
    
    Args:
        kde_grid: Values of the KDE before the update
        x_eval: The grid points
        measurement: The new data point
        n_measurements: The number of measurements in the estimate without the new data point
        bandwidth: The bandwidth of the KDE

    Returns:
        The updated values for the KDE
    """
    kernel_value = gaussian_kernel(
        x=x_eval - measurement,
        bandwidth=bandwidth
    )
    return 1 / (n_measurements + 1) * (n_measurements * kde_grid + kernel_value[..., None])
