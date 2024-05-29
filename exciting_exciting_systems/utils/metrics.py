import jax
import jax.numpy as jnp


@jax.jit
def kullback_leibler_divergence(p: jnp.ndarray, q: jnp.ndarray):
    """Computes the sample KLD between two inputs.

    The last dim of the input needs to be of length 1. The summation occurs along the second to
    last dimension. All dimensions before that are kept as they are. Overall the shape of the
    two inputs must be indentical.

    TODO: add an eps=1e-32 to remove zero issues?
    """
    assert p.shape == q.shape, "The two inputs need to be of the same shape."
    assert p.shape[-1] == q.shape[-1] == 1, "Last dim needs to be of length 1 for PDFs"

    eps = 1e-12

    kld = (p + eps) * jnp.log((p + eps) / (q + eps))
    return jnp.sum(kld, axis=-2)


@jax.jit
def KLDLoss(p: jnp.ndarray, q: jnp.ndarray):
    """Reduce mapped KLD to loss value."""
    return jnp.mean(kullback_leibler_divergence(p, q))


@jax.jit
def jensen_shannon_divergence(p: jnp.ndarray, q: jnp.ndarray):
    """Computes the sample JSD between two inputs.

    The last dim of the input needs to be of length 1. The summation occurs along the second to
    last dimension. All dimensions before that are kept as they are. Overall the shape of the
    two inputs must be indentical.
    """
    assert p.shape == q.shape, "The two inputs need to be of the same shape."
    assert p.shape[-1] == q.shape[-1] == 1, "Last dim needs to be of length 1 for PDFs"

    return (kullback_leibler_divergence(p, q) + kullback_leibler_divergence(q, p)) / 2


@jax.jit
def JSDLoss(p: jnp.ndarray, q: jnp.ndarray):
    """Reduce mapped JSD to loss value."""
    return jnp.mean(jensen_shannon_divergence(p, q))


def MNNS_without_penalty(data_points: jnp.ndarray, new_data_points: jnp.ndarray) -> jnp.ndarray:
    """From [Smits+Nelles2024].

    Implementation inspired by https://github.com/google/jax/discussions/9813

    TODO: Not sure about this penalty. Seems difficult to use for continuous action-spaces?
    They used quantized amplitude levels in their implementation.
    """
    L = new_data_points.shape[0]
    distance_matrix = jnp.linalg.norm(data_points[:, None, :] - new_data_points[None, ...], axis=-1)
    minimal_distances = jnp.min(distance_matrix, axis=0)
    return -jnp.sum(minimal_distances) / L


def audze_eglais(data_points: jnp.ndarray) -> jnp.ndarray:
    """From [Smits+Nelles2024]. The maximin-desing penalizes points that
    are too close in the point distribution.

    TODO: There has to be a more efficient way to do this.
    """
    N = data_points.shape[0]
    distance_matrix = jnp.linalg.norm(data_points[:, None, :] - data_points[None, ...], axis=-1)
    distances = distance_matrix[jax.numpy.triu_indices(N, k=1)]

    return 2 / (N * (N - 1)) * jnp.sum(1 / distances**2)


def MC_uniform_sampling_distribution_approximation(
    data_points: jnp.ndarray, support_points: jnp.ndarray
) -> jnp.ndarray:
    """From [Smits+Nelles2024]. The minimax-design tries to minimize
    the distances of the data points to the support points.

    What stops the data points to just flock to a single support point?
    This is just looking at the shortest distance.
    """
    M = support_points.shape[0]
    distance_matrix = jnp.linalg.norm(data_points[:, None, :] - support_points[None, ...], axis=-1)
    minimal_distances = jnp.min(distance_matrix, axis=0)
    return jnp.sum(minimal_distances) / M
