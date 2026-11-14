import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")

    d = np.maximum(distance_matrix, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    positive_values = finite_values[finite_values > 0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)

    inverse_distance = 1.0 / (1.0 + d / scale)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    order = np.argsort(d, axis=1, kind="mergesort")
    rank = np.empty_like(order, dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank = np.maximum(rank - 1.0, 0.0)

    rank_affinity = np.exp(-rank / max(2.0, np.sqrt(n)))
    rank_affinity[~np.isfinite(rank_affinity)] = 0.0

    nearest = order[:, 0]
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    local_scale = np.partition(d, 2, axis=1)[:, 1]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        scale,
    )
    local_affinity = 1.0 / (1.0 + d / np.sqrt(local_scale[:, None] * local_scale[None, :]))

    score = (
        0.50 * inverse_distance
        + 0.25 * 0.5 * (rank_affinity + rank_affinity.T)
        + 0.15 * local_affinity
        + 0.10 * mutual
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
