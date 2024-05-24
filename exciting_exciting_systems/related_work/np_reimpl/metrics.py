"""Re-implementation of metrics using pure numpy instead of jax.

This is necessary for a pure numpy implementation of the related work (which cannot be efficiently implemented in jax).
"""

import numpy as np


def MNNS_without_penalty(
        data_points: np.ndarray,
        new_data_points: np.ndarray
) -> np.ndarray:
    """From [Smits+Nelles2024].

    Implementation inspired by https://github.com/google/jax/discussions/9813

    TODO: Not sure about this penalty. Seems difficult to use for continuous action-spaces?
    They used quantized amplitude levels in their implementation.
    """
    L = new_data_points.shape[0]
    distance_matrix = np.linalg.norm(data_points[:, None, :] - new_data_points[None, ...], axis=-1)
    minimal_distances = np.min(distance_matrix, axis=0)
    return - np.sum(minimal_distances) / L


def average_euclidean_distance(
        data_points: np.ndarray
):
    """From [Smits+Nelles2024].
    
    Used within the MNNS penalty term.
    """
    N = data_points.shape[0]
    distance_matrix = np.linalg.norm(data_points[:, None, :] - data_points[None, ...], axis=-1)
    distances = distance_matrix[np.triu_indices(N, k=1)]
    return 2 / (N * (N-1)) * np.sum(distances**2)


def MNNS(
        data_points: np.ndarray,
        new_data_points: np.ndarray,
        k_d_max = 1,
        delta = None
):
    """From [Smits+Nelles2024].

    Implementation inspired by https://github.com/google/jax/discussions/9813

    Maximum nearest neighbor sequence. If used during optimization, the delta should
    be precomputed for the data_points.
    """    
    score_without_penalty = MNNS_without_penalty(data_points, new_data_points)

    if delta is None:
        delta = average_euclidean_distance(data_points)

    action_counts = None  
    penalty = action_counts * k_d_max * delta

    return score_without_penalty + penalty
