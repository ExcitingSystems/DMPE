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
