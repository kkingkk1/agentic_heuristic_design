import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge desirability indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d > 0)
    valid &= valid.T

    if not np.any(valid):
        return result

    symmetric_d = np.zeros_like(d, dtype=float)
    symmetric_d[valid] = 0.5 * (d[valid] + d.T[valid])

    positive = symmetric_d[valid]
    scale = np.median(positive)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.mean(positive)
    scale = max(float(scale), np.finfo(float).eps)

    normalized = np.full_like(symmetric_d, np.inf, dtype=float)
    normalized[valid] = symmetric_d[valid] / scale

    inverse = np.zeros_like(symmetric_d, dtype=float)
    inverse[valid] = 1.0 / np.maximum(normalized[valid], 1e-12)

    row_median = np.full(n, scale, dtype=float)
    for i in range(n):
        row_values = symmetric_d[i, valid[i]]
        if row_values.size:
            row_median[i] = max(float(np.median(row_values)), np.finfo(float).eps)

    local_distance = np.zeros_like(symmetric_d, dtype=float)
    local_distance[valid] = symmetric_d[valid] / np.sqrt(
        row_median[:, None][valid] * row_median[None, :][valid]
    )

    locality = np.zeros_like(symmetric_d, dtype=float)
    locality[valid] = np.exp(-np.clip(local_distance[valid], 0.0, 50.0))

    ranks = np.full((n, n), n + 1.0, dtype=float)
    nearest = np.full(n, -1, dtype=int)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = candidates[np.argsort(symmetric_d[i, candidates], kind="stable")]
            ranks[i, order] = np.arange(1, order.size + 1, dtype=float)
            nearest[i] = order[0]

    rank_score = np.zeros_like(symmetric_d, dtype=float)
    rank_score[valid] = 1.0 / np.sqrt(ranks[valid])

    mutual = np.zeros_like(symmetric_d, dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0
    mutual = np.maximum(mutual, mutual.T)

    neighbor_count = max(1, min(5, n - 1))
    neighborhoods = np.zeros((n, n), dtype=bool)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = candidates[np.argsort(symmetric_d[i, candidates], kind="stable")]
            neighborhoods[i, order[:neighbor_count]] = True

    shared = np.zeros_like(symmetric_d, dtype=float)
    for i in range(n):
        for j in np.flatnonzero(valid[i]):
            union = np.count_nonzero(neighborhoods[i] | neighborhoods[j])
            if union:
                shared[i, j] = np.count_nonzero(
                    neighborhoods[i] & neighborhoods[j]
                ) / union

    row_sum = inverse.sum(axis=1, keepdims=True)
    row_preference = inverse / np.maximum(row_sum, np.finfo(float).eps)
    row_preference = 0.5 * (row_preference + row_preference.T)

    score = (
        0.35 * inverse
        + 0.25 * locality
        + 0.18 * rank_score
        + 0.10 * row_preference
        + 0.07 * shared
        + 0.05 * mutual
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
