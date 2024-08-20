import jax
import jax.numpy as jnp


@jax.jit
def KLDLoss(p: jnp.ndarray, q: jnp.ndarray):
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
    return jnp.squeeze(jnp.sum(kld, axis=-2))


@jax.jit
def JSDLoss(p: jnp.ndarray, q: jnp.ndarray):
    """Computes the sample JSD between two inputs.

    The last dim of the input needs to be of length 1. The summation occurs along the second to
    last dimension. All dimensions before that are kept as they are. Overall the shape of the
    two inputs must be indentical.
    """
    assert p.shape == q.shape, "The two inputs need to be of the same shape."
    assert p.shape[-1] == q.shape[-1] == 1, "Last dim needs to be of length 1 for PDFs"

    m = (p + q) / 2
    return jnp.squeeze((KLDLoss(p, m) + KLDLoss(q, m)) / 2)


def MNNS_without_penalty(data_points: jnp.ndarray, new_data_points: jnp.ndarray) -> jnp.ndarray:
    """From [Smits2024].

    Implementation inspired by https://github.com/google/jax/discussions/9813

    TODO: Not sure about this penalty. Seems difficult to use for continuous action-spaces?
    They used quantized amplitude levels in their implementation.
    """
    L = new_data_points.shape[0]
    distance_matrix = jnp.linalg.norm(data_points[:, None, :] - new_data_points[None, ...], axis=-1)
    minimal_distances = jnp.min(distance_matrix, axis=0)
    return -jnp.sum(minimal_distances) / L


def audze_eglais(data_points: jnp.ndarray, eps: float = 0.001) -> jnp.ndarray:
    """From [Smits2024]. The maximin-desing penalizes points that
    are too close in the point distribution.

    TODO: There has to be a more efficient way to do this.
    """
    N = data_points.shape[0]
    distance_matrix = jnp.linalg.norm(data_points[:, None, :] - data_points[None, ...], axis=-1)
    distances = distance_matrix[jax.numpy.triu_indices(N, k=1)]

    return 2 / (N * (N - 1)) * jnp.sum(1 / (distances**2 + eps))


@jax.jit
def MC_uniform_sampling_distribution_approximation(
    data_points: jnp.ndarray, support_points: jnp.ndarray
) -> jnp.ndarray:
    """From [Smits2024]. The minimax-design tries to minimize
    the distances of the data points to the support points.

    What stops the data points to just flock to a single support point?
    This is just looking at the shortest distance.
    """
    M = support_points.shape[0]
    distance_matrix = jnp.linalg.norm(data_points[:, None, :] - support_points[None, ...], axis=-1)
    minimal_distances = jnp.min(distance_matrix, axis=0)

    return jnp.sum(minimal_distances) / M


def blockwise_mcudsa(data_points: jnp.ndarray, support_points: jnp.ndarray) -> jnp.ndarray:
    M = support_points.shape[0]

    block_size = 1_000
    value = jnp.zeros(1)

    for m in range(0, M, block_size):
        end = min(m + block_size, M)  # next block or until the end
        value = value + (
            MC_uniform_sampling_distribution_approximation(
                data_points=data_points,
                support_points=support_points[m : min(m + block_size, M)],
            )
            * (end - m)  # denormalizing mean inside loss computation
            / M
        )

    return value


@jax.jit
def kiss_space_filling_cost(
    data_points: jnp.ndarray,
    support_points: jnp.ndarray,
    variances: jnp.ndarray,
    eps: float = 1e-16,
) -> jnp.ndarray:
    """From [Kiss2024]. Slightly modified to use the mean instead of the sum in the denominator.
    The goal is to have the same metric value for identical data distributions with different number
    of data points.
    """
    difference = data_points[None, ...] - support_points[:, None, :]
    exponent = -0.5 * jnp.sum(difference**2 * 1 / variances, axis=-1)

    denominator = eps + jnp.mean(jnp.exp(exponent), axis=-1)

    return jnp.mean(1 / denominator, axis=0)


def blockwise_ksfc(
    data_points: jnp.ndarray,
    support_points: jnp.ndarray,
    variances: jnp.ndarray,
    eps: float = 1e-16,
) -> jnp.ndarray:
    M = support_points.shape[0]

    block_size = 1_000
    value = jnp.zeros(1)

    for m in range(0, M, block_size):
        end = min(m + block_size, M)  # next block or until the end
        value = value + (
            kiss_space_filling_cost(
                data_points=data_points,
                support_points=support_points[m:end],
                variances=variances,
                eps=eps,
            )
            * (end - m)  # denormalizing mean inside loss computation
            / M
        )

    return value
