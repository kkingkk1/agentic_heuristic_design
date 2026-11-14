import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distances must be non-negative")

    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_positive = d[np.isfinite(d) & (d > 0)]
    if finite_positive.size:
        eps = max(
            np.finfo(float).eps,
            float(np.median(finite_positive)) * 1e-12,
        )
    else:
        eps = 1e-12

    sorted_distances = np.sort(d, axis=1)
    neighbor_count = min(max(2, int(np.sqrt(n))), n - 1)

    local_scale = np.mean(
        sorted_distances[:, :neighbor_count],
        axis=1,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = d / np.maximum(pair_scale, eps)

    local_closeness = np.exp(
        -np.minimum(normalized_distance, 50.0)
    )

    reciprocal = 1.0 / np.maximum(d, eps)
    reciprocal[~np.isfinite(reciprocal)] = 0.0

    row_max = np.max(reciprocal, axis=1, keepdims=True)
    row_max = np.maximum(row_max, eps)
    endpoint_preference = reciprocal / row_max

    mutual_preference = np.sqrt(
        endpoint_preference * endpoint_preference.T
    )

    ranks = np.argsort(np.argsort(d, axis=1), axis=1).astype(float)
    rank_decay = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_decay * rank_decay.T)

    nearest = np.argmin(d, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest[i]
        if nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    score = (
        0.42 * local_closeness
        + 0.33 * mutual_preference
        + 0.20 * mutual_rank
        + 0.05 * mutual_nearest
    )

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
