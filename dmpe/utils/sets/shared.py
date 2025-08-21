from typing import Callable
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
        return check_in_set(obs, self.mask, self.grid)

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

        if isinstance(self.unflattened_shape, list):
            assert self.unflattened_shape == other.unflattened_shape, (
                "The unflattened shape of the Sets must be the same.",
            )

        return DiscretizedSet(
            grid=self.grid,
            mask=jnp.logical_and(self.mask, other.mask),
            unflattened_shape=self.unflattened_shape,
        )

    def visualize(
        self, reduction_method: Callable = jnp.sum, labels: None | list[str] = None, use_contourf: bool = True
    ):
        if len(self.unflattened_shape) == 1:
            fig, axs = plt.subplots(1, 1, figsize=(9, 9))
            axs.plot(self.grid, self.mask)
        elif len(self.unflattened_shape) == 2:
            fig, axs = plt.subplots(1, 1, figsize=(9, 9))
            axs.contourf(
                self.grid_unflattened[..., 0],
                self.grid_unflattened[..., 1],
                self.mask_unflattened,
            )
            return fig, axs
        else:
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

                    reduced_safe = reduction_method(self.mask_unflattened, axis=tuple(reduction_indices))

                    if i > j:
                        reduced_safe = jnp.transpose(reduced_safe)

                    if use_contourf:
                        axs[j, i].contourf(
                            self.grid_unflattened[..., *[0 for _ in range(dim - 2)], 0],
                            self.grid_unflattened[..., *[0 for _ in range(dim - 2)], 1],
                            reduced_safe,
                        )
                    else:
                        axs[j, i].imshow(reduced_safe.T, origin="lower", extent=[-1, 1, -1, 1])
                    axs[j, 0].set_ylabel(labels[j])

                axs[-1, i].set_xlabel(labels[i])
            fig.tight_layout()
            return fig, axs


class SlicedSet(eqx.Module):
    sets: list[DiscretizedSet]

    def check_in_set(self, obs: jax.Array, reduce: bool = True) -> jax.Array:
        check_list = jnp.array([s.check_in_set(obs) for s in self.sets])
        if reduce:
            return jnp.all(check_list)
        else:
            return check_list


def save_discretized_set(filename: str, set: DiscretizedSet):
    data = dict(
        grid=set.grid.tolist(),
        mask=set.mask.tolist(),
        unflattened_shape=set.unflattened_shape,
    )
    with open(filename, "w") as f:
        json.dump(data, f)


def load_discretized_set(filename: str) -> DiscretizedSet:
    with open(filename, "r") as f:
        data = json.load(f)
    return DiscretizedSet(jnp.array(data["grid"]), jnp.array(data["mask"]), tuple(data["unflattened_shape"]))
