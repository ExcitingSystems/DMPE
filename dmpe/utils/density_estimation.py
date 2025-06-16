from typing import Callable

import jax
import jax.numpy as jnp

import equinox as eqx


def select_bandwidth(
    delta_z: float,
    dim: int,
    n_g: int,
    percentage: float,
) -> float:
    """Select a bandwidth for the kernel density estimate by a rough heuristic.

    The bandwidth is designed so that the kernel is still at a given percentage of
    its maximum value when a step is taken in each dimension of the underlying
    grid.

    Args:
        delta_z (float): The size of the space in each dimension. Assumed to be
            symmetrical
        dim (int): The dimension of the space.
        n_g (int): Number of grid points per dimension.
        percentage (float): The percentage of the maximum value of the kernel at
            the other grid point reached by stepping once in each dimension on
            the grid.
    Returns:
        bandwidth (float): The resulting proposed bandwidth
    """
    return delta_z * jnp.sqrt(dim) / (n_g * jnp.sqrt(-2 * jnp.log(percentage)))


@jax.jit
def gaussian_kernel(z: jax.Array, bandwidth: float) -> jax.Array:
    """Evaluates the Gaussian RBF kernel at z with given bandwidth. This can take arbitrary
    dimensions for 'z' and will compute the output by broadcasting. The last dimension of
    the input needs to be the dimension of the data which is reduced along.

    Args:
        z (jax.Array): The input data points with shape (..., data_dim). The first dimensions
            are broadcast over and the last dimension is assumed to be the feature/data-dimension.
        bandwidth (float): Bandwidth of the kernel. Always symmetrical in the feature space.

    Returns:
        The kernel value as a jax.Array with shape (..., 1). Where all dimensions before the
        last one are kept as they are.
    """
    data_dim = z.shape[-1]
    factor = bandwidth**data_dim * jnp.power(2 * jnp.pi, data_dim / 2)
    return 1 / factor * jnp.exp(-jnp.linalg.norm(z, axis=-1) ** 2 / (2 * bandwidth**2))


class DensityEstimate(eqx.Module):
    """Holds an estimate of the data density of sampled data points.

    Args:
        p (jax.Array): The probability estimates at the grid points
        z_g (jax.Array): The grid points to which the probability estimates belong
        bandwidth (jax.Array): The bandwidth of the kernel density estimate
        n_observations (jax.Array): The number of observations that make up the momentary
            estimate
    """

    p: jax.Array
    z_g: jax.Array
    bandwidth: jax.Array
    n_observations: jax.Array

    @classmethod
    def from_estimate(
        cls,
        p: jax.Array,
        n_additional_observations: int,
        density_estimate: "DensityEstimate",
    ) -> "DensityEstimate":
        """Create a density estimate from an existing estimate.

        The computation is usually that p is constructed from the DensityEstimate.p before the
        update together with the kernel density estimation over new data points. The result of
        this computation is the input 'p'.

        Args:
            p (jax.Array): The new probability estimate
            n_additional_observations (int): The number of data points that have been added
                in the update
            density_estimate (DensityEstimate): The density estimate before the update

        Returns:
            The updated density estimate (DensityEstimate)
        """

        return cls(
            p=p,
            n_observations=(density_estimate.n_observations + n_additional_observations),
            z_g=density_estimate.z_g,
            bandwidth=density_estimate.bandwidth,
        )

    @classmethod
    def from_observations_actions(
        cls,
        observations: jax.Array,
        actions: jax.Array,
        use_actions: bool = True,
        points_per_dim: int = 30,
        z_min: float = -1,
        z_max: float = 1,
        bandwidth: float = 0.05,
    ) -> "DensityEstimate":
        """Create a fresh density estimate from gathered in the form of observations and actions.

        It is assumed that the feature vector is made up from observations and actions together
        when 'use_actions' is True and only from the observations if it is False.

        Often, the actions will have one less element compared to the observations. This occurs
        when an observation has resulted from the application of the last action but a new action
        has not been chosen yet. The last observation is then omitted from the density estimate.

        Args:
            observations (jax.Array): The observations to be part of the density estimate
            actions (jax.Array): The actions to be part of the density estimate
            use_actions (bool): A flag indicating whether the actions are to be part of the density
                estimate
            points_per_dim (int): The number of grid points per dimension. Always identical for each
                dimension
            z_min (float): The minimum value of the grid in each dimension
            z_max (float): The maximum value of the grid in each dimension
            bandwidth (float): The bandwidth of the kernel density estimate

        Returns:
            The resulting density estimate (DensityEstimate)
        """

        if observations.shape[0] == actions.shape[0] + 1:
            data_points = (
                jnp.concatenate([observations[0:-1, :], actions], axis=-1)[None] if use_actions else observations[None]
            )
        else:
            data_points = jnp.concatenate([observations, actions], axis=-1)[None] if use_actions else observations[None]

        return cls.from_dataset(data_points, points_per_dim, z_min, z_max, bandwidth)

    @classmethod
    def from_dataset(
        cls,
        data_points: jax.Array,
        points_per_dim: int = 30,
        z_min: float = -1,
        z_max: float = 1,
        bandwidth: float = 0.05,
    ) -> "DensityEstimate":
        """Create a fresh density estimate from gathered in the form of a sequence of feature vectors.

        Args:
            data_points (jax.Array): The data points to be part of the density estimate
            points_per_dim (int): The number of grid points per dimension. Always identical for each
                dimension
            z_min (float): The minimum value of the grid in each dimension
            z_max (float): The maximum value of the grid in each dimension
            bandwidth (float): The bandwidth of the kernel density estimate

        Returns:
            The resulting density estimate (DensityEstimate)
        """
        dim = data_points.shape[-1]

        n_grid_points = points_per_dim**dim
        density_estimate = cls(
            p=jnp.zeros([1, n_grid_points, 1]),
            z_g=build_grid(dim, z_min, z_max, points_per_dim),
            bandwidth=jnp.array([bandwidth]),
            n_observations=jnp.array([0]),
        )
        density_estimate = jax.vmap(
            update_density_estimate_multiple_observations,
            in_axes=(DensityEstimate(0, None, None, None), 0),
            out_axes=(DensityEstimate(0, None, None, None)),
        )(
            density_estimate,
            data_points,
        )
        return density_estimate


@jax.jit
def update_density_estimate_single_observation(
    density_estimate: DensityEstimate,
    data_point: jax.Array,
) -> DensityEstimate:
    """Recursive update to the kernel density estimation (KDE) on a fixed grid.

    Args:
        density_estimate (DensityEstimate): The density estimate before the update
        data_point (jax.Array): The new data point

    Returns:
        The updated density estimate (DensityEstimate)
    """
    kernel_value = gaussian_kernel(z=density_estimate.z_g - data_point, bandwidth=density_estimate.bandwidth)
    p_est = (
        1
        / (density_estimate.n_observations + 1)
        * (density_estimate.n_observations * density_estimate.p + kernel_value[..., None])
    )

    return DensityEstimate.from_estimate(p=p_est, n_additional_observations=1, density_estimate=density_estimate)


@jax.jit
def update_density_estimate_multiple_observations(
    density_estimate: DensityEstimate,
    data_points: jax.Array,
) -> DensityEstimate:
    """Add a new sequence of data points to the current data density estimate.

    Args:
        density_estimate (DensityEstimate): The density estimate before the update
        data_points (jax.Array): The sequence of data_points

    Returns:
        The updated density estimate (DensityEstimate)
    """

    def shifted_gaussian_kernel(z, data_points, bandwidth):
        # created to enable vmapping of data_points without vmapping of z
        return gaussian_kernel(z - data_points, bandwidth)

    new_sum_part = jax.vmap(shifted_gaussian_kernel, in_axes=(None, 0, None))(
        density_estimate.z_g, data_points, density_estimate.bandwidth
    )
    new_sum_part = jnp.sum(new_sum_part, axis=0)[..., None]
    p_est = (
        1
        / (density_estimate.n_observations + data_points.shape[0])
        * (density_estimate.n_observations * density_estimate.p + new_sum_part)
    )

    return DensityEstimate.from_estimate(
        p=p_est, n_additional_observations=data_points.shape[0], density_estimate=density_estimate
    )


def build_grid(dim: int, low: float, high: float, points_per_dim: int) -> jax.Array:
    """Build a uniform grid of points in the given dimension.

    Args:
        dim (int): Dimensionality of the grid (and feature vector z)
        low (float): The minimum value of the grid in each dimension
        high (float): The maximum value of the grid in each dimension
        points_per_dim (int): The number of grid points per dimension. Always identical for each
            dimension

    Returns:
        The flattened grid as a jax.Array with shape (points_per_dim**dim, dim)
    """
    xs = [jnp.linspace(low, high, points_per_dim) for _ in range(dim)]

    z_g = jnp.meshgrid(*xs)
    z_g = jnp.stack([_x for _x in z_g], axis=-1)
    z_g = z_g.reshape(-1, dim)

    assert z_g.shape[0] == points_per_dim**dim
    return z_g


def build_grid_2d(low: float, high: float, points_per_dim: int):
    """Shorthand for a uniform 2d grid.

    Args:
        low (float): The minimum value of the grid in each dimension
        high (float): The maximum value of the grid in each dimension
        points_per_dim (int): The number of grid points per dimension. Always identical for each
            dimension

    Returns:
        The flattened grid as a jax.Array with shape (points_per_dim**2, 2)
    """
    return build_grid(2, low, high, points_per_dim)


def build_grid_3d(low: float, high: float, points_per_dim: int):
    """Shorthand for a uniform 3d grid.

    Args:
        low (float): The minimum value of the grid in each dimension
        high (float): The maximum value of the grid in each dimension
        points_per_dim (int): The number of grid points per dimension. Always identical for each
            dimension

    Returns:
        The flattened grid as a jax.Array with shape (points_per_dim**3, 3)
    """
    return build_grid(3, low, high, points_per_dim)


def get_uniform_target_distribution(
    dim: int,
    points_per_dim: int,
    bandwidth: float,
    grid_extend: float,
    consider_action_distribution: bool,
    penalty_function: Callable,
) -> jax.Array:
    """Get a uniform target distribution for the DMPE algorithm based on the grid parameters
    and a penalty function. Only values that are not penalized by the penalty function
    are targeted in the target distribution, but all of them are to be covered uniformly.

    Args:
        dim (int): Number of dimensions for the grid.
        points_per_dim (int): The number of grid points per dimension. Always identical for each
            dimension
        bandwidth (float): The bandwidth of the kernel density estimate
        grid_extend (float): The extent of the grid in each dimension
        consider_action_distribution (bool): A flag indicating whether the action distribution
            is to be considered
        penalty_function (Callable): The penalty function that is used to determine the
            valid grid points

    Returns:
        The target distribution as a jax.Array with shape (points_per_dim**dim, 1)

    """
    z_g = build_grid(dim, low=-grid_extend, high=grid_extend, points_per_dim=points_per_dim)

    if consider_action_distribution:
        constr_func = lambda z_g: penalty_function(z_g[..., None, :2], z_g[..., None, 2:])
    else:
        constr_func = lambda z_g: penalty_function(z_g[..., None, :2], None)

    valid_grid_point = jax.vmap(constr_func, in_axes=0)(z_g) == 0
    constrained_data_points = z_g[jnp.where(valid_grid_point == True)]

    target_distribution = DensityEstimate.from_dataset(
        constrained_data_points[None],
        z_min=-grid_extend,
        z_max=grid_extend,
        points_per_dim=points_per_dim,
        bandwidth=bandwidth,
    )
    return target_distribution.p[0] / jnp.sum(target_distribution.p[0])
