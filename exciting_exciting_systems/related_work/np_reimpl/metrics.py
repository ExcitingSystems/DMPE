"""Re-implementation of metrics using pure numpy instead of jax.

This is necessary for a pure numpy implementation of the related work (which cannot be efficiently implemented in jax).
"""

import numpy as np


def KLDLoss(p: np.ndarray, q: np.ndarray):
    """Computes the sample KLD between two inputs.

    The last dim of the input needs to be of length 1. The summation occurs along the second to
    last dimension. All dimensions before that are kept as they are. Overall the shape of the
    two inputs must be indentical.

    TODO: add an eps=1e-32 to remove zero issues?
    """
    assert p.shape == q.shape, "The two inputs need to be of the same shape."
    assert p.shape[-1] == q.shape[-1] == 1, "Last dim needs to be of length 1 for PDFs"

    eps = 1e-12

    kld = (p + eps) * np.log((p + eps) / (q + eps))
    return np.squeeze(np.sum(kld, axis=-2))


def JSDLoss(p: np.ndarray, q: np.ndarray):
    """Computes the sample JSD between two inputs.

    The last dim of the input needs to be of length 1. The summation occurs along the second to
    last dimension. All dimensions before that are kept as they are. Overall the shape of the
    two inputs must be indentical.
    """
    assert p.shape == q.shape, "The two inputs need to be of the same shape."
    assert p.shape[-1] == q.shape[-1] == 1, "Last dim needs to be of length 1 for PDFs"

    m = (p + q) / 2
    return np.squeeze((KLDLoss(p, m) + KLDLoss(q, m)) / 2)


def MNNS_without_penalty(data_points: np.ndarray, new_data_points: np.ndarray) -> np.ndarray:
    """From [Smits2024].

    Implementation inspired by https://github.com/google/jax/discussions/9813

    TODO: Not sure about this penalty. Seems difficult to use for continuous action-spaces?
    They used quantized amplitude levels in their implementation.
    """
    L = new_data_points.shape[0]
    distance_matrix = np.linalg.norm(data_points[:, None, :] - new_data_points[None, ...], axis=-1)
    minimal_distances = np.min(distance_matrix, axis=0)
    return -np.sum(minimal_distances) / L


def average_euclidean_distance(data_points: np.ndarray):
    """From [Smits2024].

    Used within the MNNS penalty term.
    """
    N = data_points.shape[0]
    distance_matrix = np.linalg.norm(data_points[:, None, :] - data_points[None, ...], axis=-1)
    distances = distance_matrix[np.triu_indices(N, k=1)]
    return 2 / (N * (N - 1)) * np.sum(distances**2)


def MNNS(data_points: np.ndarray, new_data_points: np.ndarray, k_d_max=1, delta=None):
    """From [Smits2024].

    Implementation inspired by https://github.com/google/jax/discussions/9813

    Maximum nearest neighbor sequence. If used during optimization, the delta should
    be precomputed for the data_points.
    """
    score_without_penalty = MNNS_without_penalty(data_points, new_data_points)

    if delta is None:
        delta = average_euclidean_distance(data_points)

    raise NotImplementedError
    # action_counts = None
    # penalty = action_counts * k_d_max * delta

    # return score_without_penalty + penalty


def MC_uniform_sampling_distribution_approximation(data_points: np.ndarray, support_points: np.ndarray) -> np.ndarray:
    """From [Smits2024]. The minimax-design tries to minimize
    the distances of the data points to the support points.

    What stops the data points to just flock to a single support point?
    This is just looking at the shortest distance.
    """
    M = support_points.shape[0]
    distance_matrix = np.linalg.norm(data_points[:, None, :] - support_points[None, ...], axis=-1)
    minimal_distances = np.min(distance_matrix, axis=0)
    return np.sum(minimal_distances) / M


def audze_eglais(data_points: np.ndarray, eps: float = 0.001) -> np.ndarray:
    """From [Smits2024]. The maximin-desing penalizes points that
    are too close in the point distribution.

    TODO: There has to be a more efficient way to do this.
    """
    N = data_points.shape[0]
    distance_matrix = np.linalg.norm(data_points[:, None, :] - data_points[None, ...], axis=-1)
    distances = distance_matrix[np.triu_indices(N, k=1)]

    return 2 / (N * (N - 1)) * np.sum(1 / (distances**2 + eps))


def blockwise_mcudsa(data_points: np.ndarray, support_points: np.ndarray) -> np.ndarray:
    M = support_points.shape[0]

    block_size = 1_000
    value = np.zeros(1)

    for m in range(0, M, block_size):
        end = min(m + block_size, M)  # next block or until the end
        value = value + (
            MC_uniform_sampling_distribution_approximation(
                data_points=data_points,
                support_points=support_points[m:end],
            )
            * (end - m)  # denormalizing mean inside loss computation
            / M
        )

    return value


def kiss_space_filling_cost(
    data_points: np.ndarray,
    support_points: np.ndarray,
    variances: np.ndarray,
    eps: float = 1e-16,
) -> np.ndarray:
    """From [Kiss2024]. Slightly modified to use the mean instead of the sum in the denominator.
    The goal is to have the same metric value for identical data distributions with different number
    of data points.
    """
    difference = data_points[None, ...] - support_points[:, None, :]
    exponent = -0.5 * np.sum(difference**2 * 1 / variances, axis=-1)

    denominator = eps + np.mean(np.exp(exponent), axis=-1)

    return np.mean(1 / denominator, axis=0)


def blockwise_ksfc(
    data_points: np.ndarray,
    support_points: np.ndarray,
    variances: np.ndarray,
    eps: float = 1e-16,
) -> np.ndarray:
    M = support_points.shape[0]

    block_size = 1_000
    value = np.zeros(1)

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
