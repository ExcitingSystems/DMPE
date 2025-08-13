import json
import matplotlib.pyplot as plt

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
    mask: jax.Array,
    grid: jax.Array,
) -> jax.Array:
    """Check if the given observation is within the set.

    Args:
        obs (jax.Array): The observation to be tested with shape (obs_dim,)
        mask (jax.Array): The boolean array describing which grid points belong to the set with shape (points_per_dim**dim,)
        grid (jax.Array): The float array describing the positions of the grid points with shape (points_per_dim**dim, obs_dim)
        penalty_function (Callable): Penalty function for the observation constraints

    Returns:
        A boolean jax.Array indicating if the input belongs to the set.
    """
    dist = jnp.linalg.norm(obs[None] - grid, axis=-1)
    min_idx = jnp.argmin(dist)

    penalty_value = soft_penalty(obs[None])
    penalty_bool = jnp.isclose(penalty_value, 0)

    return jnp.logical_and(mask[min_idx], penalty_bool)


class DiscretizedSet(eqx.Module):
    grid: jax.Array
    mask: jax.Array
    unflattened_shape: tuple[int] = eqx.field(static=True)

    def check_in_set(self, obs: jax.Array) -> jax.Array:
        return check_in_set(obs, self.mask, self.set_grid)

    @property
    def mask_unflattened(self):
        return self.mask.reshape(self.unflattened_shape)

    @property
    def grid_unflattened(self):
        return self.grid.reshape(list(self.unflattened_shape) + [-1])

    @property
    def unflattened(self):
        return self.mask_unflattened, self.grid_unflattened

    def __and__(self, other):
        assert jnp.all(self.grid == other.grid), "The Sets have to be discretized on the same grid."
        assert self.unflattened_shape == other.unflattened_shape, (
            "The unflattened shape of the Sets must be the same.",
        )

        return DiscretizedSet(
            grid=self.grid,
            mask=jnp.logical_and(self.mask, other.mask),
            unflattened_shape=self.unflattened_shape,
        )

    def visualize(self, labels=None):
        if self.grid.shape[-1] == 2:
            plt.contourf(
                self.grid_unflattened[..., 0],
                self.grid_unflattened[..., 1],
                self.mask_unflattened,
            )
            plt.show()

        elif self.grid.shape[-1] < 5:
            dim = self.grid.shape[-1]
            fig, axs = plt.subplots(nrows=dim, ncols=dim, figsize=(9, 9), sharex=True, sharey=True)
            feature_indices = jnp.arange(0, dim, 1).tolist()

            if labels is None:
                labels = jnp.arange(0, dim, 1).tolist()

            for i in range(dim):
                for j in range(dim):

                    axs[j, i].grid(True)

                    reduction_indices = [f_idx for f_idx in feature_indices if not (f_idx == i or f_idx == j)]
                    if len(reduction_indices) == dim - 1:
                        continue

                    any_safe = jnp.any(self.mask, axis=tuple(reduction_indices))

                    if i < j:
                        any_safe = jnp.transpose(any_safe)

                    axs[j, i].contourf(
                        self.grid_unflattened[..., 0],
                        self.grid_unflattened[..., 1],
                        any_safe,
                    )

                    axs[j, 0].set_ylabel(labels[j])

                axs[-1, i].set_xlabel(labels[i])
            fig.tight_layout()
        else:
            raise NotImplementedError()
