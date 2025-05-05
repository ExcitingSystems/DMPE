import jax.numpy as jnp

from dmpe.utils.density_estimation import (
    DensityEstimate,
    build_grid,
    update_density_estimate_multiple_observations,
)
from dmpe.utils.metrics import (
    JSDLoss,
    audze_eglais,
    MC_uniform_sampling_distribution_approximation,
    kiss_space_filling_cost,
    blockwise_ksfc,
    blockwise_mcudsa,
)


def default_jsd(
    observations, actions, points_per_dim=50, bounds=(-1, 1), bandwidth=0.05, target_distribution=None, ca=True
):

    if observations.shape[0] == actions.shape[0] + 1:
        observations = observations[0:-1, :]

    if ca:
        data_points = jnp.concatenate([observations, actions], axis=-1)
    else:
        data_points = observations

    dim = data_points.shape[-1]
    n_grid_points = points_per_dim**dim

    density_estimate = DensityEstimate(
        p=jnp.zeros([n_grid_points, 1]),
        z_g=build_grid(dim, low=bounds[0], high=bounds[1], points_per_dim=points_per_dim),
        bandwidth=jnp.array([bandwidth]),
        n_observations=jnp.array([0]),
    )

    if data_points.shape[0] > 500:
        # if there are too many datapoints at once, split them up and add
        # them in smaller chunks to the density estimate

        block_size = 500

        for n in range(0, data_points.shape[0] + 1, block_size):
            density_estimate = update_density_estimate_multiple_observations(
                density_estimate,
                data_points[n : min(n + block_size, data_points.shape[0])],
            )
    else:
        density_estimate = update_density_estimate_multiple_observations(
            density_estimate,
            data_points,
        )

    if target_distribution is None:
        target_distribution = jnp.ones(density_estimate.p.shape)
        target_distribution /= jnp.sum(target_distribution)

    return JSDLoss(
        p=density_estimate.p / jnp.sum(density_estimate.p),
        q=target_distribution,
    )


def default_ae(observations, actions):
    if observations.shape[0] == actions.shape[0] + 1:
        observations = observations[0:-1, :]

    return audze_eglais(jnp.concatenate([observations, actions], axis=-1))


def default_mcudsa(observations, actions, bounds=(-1, 1), points_per_dim=20, ca=True):
    if observations.shape[0] == actions.shape[0] + 1:
        observations = observations[0:-1, :]

    if ca:
        data_points = jnp.concatenate([observations, actions], axis=-1)
    else:
        data_points = observations
    dim = data_points.shape[-1]

    support_points = build_grid(dim, low=bounds[0], high=bounds[1], points_per_dim=points_per_dim)

    if dim > 2:
        return blockwise_mcudsa(data_points=data_points, support_points=support_points)
    else:
        return MC_uniform_sampling_distribution_approximation(data_points=data_points, support_points=support_points)


def default_ksfc(observations, actions, points_per_dim=20, bounds=(-1, 1), variance=0.1, eps=1e-12, ca=True):
    if observations.shape[0] == actions.shape[0] + 1:
        observations = observations[0:-1, :]

    if ca:
        data_points = jnp.concatenate([observations, actions], axis=-1)
    else:
        data_points = observations
    dim = data_points.shape[-1]

    support_points = build_grid(dim, low=bounds[0], high=bounds[1], points_per_dim=points_per_dim)

    if dim > 2:
        return blockwise_ksfc(
            data_points=data_points, support_points=support_points, variances=jnp.ones([dim]) * variance, eps=eps
        )
    else:
        return kiss_space_filling_cost(
            data_points=data_points, support_points=support_points, variances=jnp.ones([dim]) * variance, eps=eps
        )
