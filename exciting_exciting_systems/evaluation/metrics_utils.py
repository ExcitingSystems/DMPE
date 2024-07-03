import jax
import jax.numpy as jnp

from exciting_exciting_systems.utils.density_estimation import (
    DensityEstimate,
    build_grid,
    update_density_estimate_multiple_observations,
)
from exciting_exciting_systems.utils.metrics import (
    JSDLoss,
    audze_eglais,
    MC_uniform_sampling_distribution_approximation,
)


def default_jsd(observations, actions, points_per_dim=50, bounds=(-1, 1), bandwidth=0.05):

    if observations.shape[0] == actions.shape[0] + 1:
        observations = observations[0:-1, :]

    data_points = jnp.concatenate([observations, actions], axis=-1)
    dim = data_points.shape[-1]
    n_grid_points = points_per_dim**dim

    density_estimate = DensityEstimate(
        p=jnp.zeros([n_grid_points, 1]),
        x_g=build_grid(dim, low=bounds[0], high=bounds[1], points_per_dim=points_per_dim),
        bandwidth=jnp.array([bandwidth]),
        n_observations=jnp.array([0]),
    )

    density_estimate = update_density_estimate_multiple_observations(
        density_estimate,
        data_points,
    )

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


def default_mcudsa(observations, actions, bounds=(-1, 1), points_per_dim=20):
    if observations.shape[0] == actions.shape[0] + 1:
        observations = observations[0:-1, :]

    data_points = jnp.concatenate([observations, actions], axis=-1)
    dim = data_points.shape[-1]

    support_points = build_grid(dim, low=bounds[0], high=bounds[1], points_per_dim=points_per_dim)

    return MC_uniform_sampling_distribution_approximation(data_points=data_points, support_points=support_points)
