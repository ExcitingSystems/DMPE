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
        observation: jnp.ndarray,
        n_observations: int,
        bandwidth: float,
    ) -> jnp.ndarray:
    """Recursive update to the kernel density estimation (KDE) on a fixed grid.
    
    Args:
        kde_grid: Values of the KDE before the update
        x_eval: The grid points
        observation: The new data point
        n_observations: The number of observations in the estimate without the new data point
        bandwidth: The bandwidth of the KDE

    Returns:
        The updated values for the KDE
    """
    kernel_value = gaussian_kernel(
        x=x_eval - observation,
        bandwidth=bandwidth
    )
    return 1 / (n_observations + 1) * (n_observations * kde_grid + kernel_value[..., None])


# @partial(jax.jit, static_argnums=(0))
# def update_kde_grid_multiple_observations(
#         n_steps: int,
#         p_est: jnp.ndarray,
#         x_eval: jnp.ndarray,
#         observations,
#         start_n_observations,
#         bandwidth: float
#     ) -> jnp.ndarray:

#     def body_fun(n, p_est):
#         p_est = update_kde_grid(
#             p_est,
#             x_eval[None, ...],
#             observation=observations[:, n, :][:, None, :],
#             n_observations=n + start_n_observations,
#             bandwidth=bandwidth
#         )

#         return p_est

#     p_est = jax.lax.fori_loop(lower=0, upper=n_steps, body_fun=body_fun, init_val=(p_est))
#     return p_est


@jax.jit
def update_kde_grid_multiple_observations(
        p_est: jnp.ndarray,
        x_eval: jnp.ndarray,
        observations: jnp.ndarray,
        n_observations: int,
        bandwidth: float
    ) -> jnp.ndarray:
    """Add a new sequence of observations to the current data density estimate.
    
    Args:
        p_est: Values of the density estimate before the update
        x_eval: The grid points of the density estimate
        observations: The sequence of observations
        n_observations: The number of observations in the estimate without the new data observations
        bandwidth: The bandwidth of the KDE
    
    Returns:
        The updated values for the density estimate
    """

    shifted_gaussian_kernel = lambda x, observation, bandwidth : gaussian_kernel(x - observation, bandwidth)
    new_sum_part = jax.vmap(shifted_gaussian_kernel, in_axes=(None, 1, None))(x_eval, observations, bandwidth)
    new_sum_part = jnp.sum(new_sum_part, axis=0)[None, ..., None]
    p_est = 1 / (n_observations + observations.shape[1]) * (n_observations * p_est + new_sum_part)

    return p_est
