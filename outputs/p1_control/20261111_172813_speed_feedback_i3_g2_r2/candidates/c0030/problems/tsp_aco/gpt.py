import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Generate normalized edge-priority indicators for stochastic TSP sampling.
    Larger values indicate more promising edges.
    """

    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = (
        np.isfinite(distance_matrix)
        & (distance_matrix >= 0.0)
        & ~diagonal
    )

    positive = distance_matrix[
        np.isfinite(distance_matrix) & (distance_matrix > 0.0)
    ]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    epsilon = np.finfo(float).eps
    fallback = max(scale, epsilon) * 1e6

    safe_distance = np.where(valid, distance_matrix, fallback)
    safe_distance = np.maximum(safe_distance, 0.0)
    safe_distance[diagonal] = np.inf

    inverse_distance = np.zeros((n, n), dtype=float)
    inverse_distance[valid] = 1.0 / np.maximum(
        safe_distance[valid], epsilon
    )

    row_max = np.max(inverse_distance, axis=1, keepdims=True)
    row_max[row_max <= 0.0] = 1.0
    local_preference = inverse_distance / row_max

    order = np.argsort(-local_preference, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values

    rank_weight = 1.0 / np.log2(ranks + 1.0)
    rank_weight[~valid] = 0.0

    reciprocal_affinity = np.sqrt(
        np.maximum(local_preference, 0.0)
        * np.maximum(local_preference.T, 0.0)
    )

    rank_affinity = np.sqrt(
        np.maximum(rank_weight, 0.0)
        * np.maximum(rank_weight.T, 0.0)
    )

    bidirectional_affinity = reciprocal_affinity * rank_affinity
    bidirectional_affinity[~valid] = 0.0

    nearest = np.argmin(safe_distance, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest[i]
        if i != j and valid[i, j] and nearest[j] == i:
            mutual_nearest[i, j] = 1.0

    nearest_distance = np.min(safe_distance, axis=1)
    savings = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if valid[i, j]:
                baseline = nearest_distance[i] + nearest_distance[j]
                savings[i, j] = max(
                    0.0,
                    baseline - safe_distance[i, j],
                )

    savings_max = np.max(savings)
    if savings_max > 0.0:
        savings /= savings_max

    reciprocal_inverse = 0.5 * (
        local_preference + local_preference.T
    )

    score = (
        0.50 * bidirectional_affinity
        + 0.25 * reciprocal_inverse
        + 0.15 * savings
        + 0.10 * mutual_nearest
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
