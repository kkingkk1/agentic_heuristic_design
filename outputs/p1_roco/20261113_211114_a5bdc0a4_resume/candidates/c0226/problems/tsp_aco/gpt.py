import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_distance = 0.5 * (distances + distances.T)
    symmetric_distance[~valid] = np.inf
    np.fill_diagonal(symmetric_distance, np.inf)

    finite_values = symmetric_distance[np.isfinite(symmetric_distance)]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    eps = np.finfo(float).eps * max(global_scale, 1.0)
    finite_distance = np.where(np.isfinite(symmetric_distance), symmetric_distance, 0.0)

    positive_count = max(1, min(8, n - 1))
    local_scales = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row = symmetric_distance[i]
        candidates = row[np.isfinite(row) & (row > 0.0)]
        if candidates.size:
            candidates.sort()
            local_scales[i] = float(
                np.median(candidates[:positive_count])
            )

    local_scales = np.maximum(local_scales, eps)
    pair_scale = np.sqrt(
        np.maximum(local_scales[:, None] * local_scales[None, :], eps)
    )

    robust_distance = symmetric_distance / max(global_scale, eps)
    local_distance = symmetric_distance / pair_scale

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (1.0 + robust_distance[valid])

    local_affinity = np.zeros((n, n), dtype=float)
    local_affinity[valid] = np.exp(
        -np.minimum(local_distance[valid], 50.0)
    )

    global_affinity = np.zeros((n, n), dtype=float)
    global_affinity[valid] = np.exp(
        -np.minimum(robust_distance[valid], 50.0)
    )

    row_max = np.max(
        np.where(valid, inverse, 0.0),
        axis=1,
        keepdims=True,
    )
    relative_closeness = inverse / np.maximum(row_max, eps)

    local_mass = local_affinity.sum(axis=1)
    local_probability = local_affinity / np.maximum(local_mass[:, None], eps)

    mutual_preference = np.sqrt(
        np.maximum(local_probability * local_probability.T, 0.0)
    )

    nearest_rank = np.zeros((n, n), dtype=float)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[np.argsort(symmetric_distance[i, indices])]
            ranks = np.arange(1.0, order.size + 1.0)
            nearest_rank[i, order] = 1.0 / np.sqrt(ranks)

    rank_mutuality = np.sqrt(
        np.maximum(nearest_rank * nearest_rank.T, 0.0)
    )

    scarcity = 1.0 / np.sqrt(np.maximum(local_mass, eps))
    scarcity /= max(float(np.median(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.6, 1.8)

    endpoint_scarcity = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )

    score = inverse
    score *= 0.55 + 0.45 * relative_closeness
    score *= 0.70 + 0.55 * mutual_preference
    score *= 0.75 + 0.45 * rank_mutuality
    score *= 0.80 + 0.30 * endpoint_scarcity
    score *= 0.85 + 0.15 * local_affinity
    score *= 0.90 + 0.10 * global_affinity

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return score.astype(float, copy=False)
