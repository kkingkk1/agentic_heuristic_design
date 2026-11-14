import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances) & (distances >= 0.0)
    valid = finite & finite.T
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_distance = 0.5 * (distances + distances.T)
    symmetric_distance[~valid] = np.inf
    np.fill_diagonal(symmetric_distance, np.inf)

    finite_values = symmetric_distance[np.isfinite(symmetric_distance)]
    positive_values = finite_values[finite_values > 0.0]

    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else float(np.median(finite_values))
        if finite_values.size
        else 1.0
    )
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)
    safe_distance = np.where(np.isfinite(symmetric_distance), symmetric_distance, 0.0)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (safe_distance[valid] + eps)

    row_max = np.max(inverse, axis=1, keepdims=True)
    relative_inverse = inverse / np.maximum(row_max, eps)

    ranks = np.zeros((n, n), dtype=float)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[np.argsort(safe_distance[i, indices], kind="stable")]
            ranks[i, order] = np.arange(1.0, order.size + 1.0)

    degree = valid.sum(axis=1).astype(float)
    rank_score = np.zeros((n, n), dtype=float)
    rank_score[valid] = 1.0 / np.sqrt(
        np.maximum(ranks[valid], 1.0)
    )
    rank_score /= np.maximum(
        np.max(rank_score, axis=1, keepdims=True),
        eps,
    )

    local_scale = np.zeros(n, dtype=float)
    for i in range(n):
        values = safe_distance[i, valid[i]]
        if values.size:
            local_scale[i] = np.median(values)
    fallback = float(np.median(local_scale[local_scale > 0.0])) \
        if np.any(local_scale > 0.0) else scale
    local_scale = np.where(local_scale > 0.0, local_scale, fallback)

    temperature = np.sqrt(np.maximum(local_scale[:, None] * local_scale[None, :], eps))
    soft = np.zeros_like(inverse)
    soft[valid] = np.exp(
        -np.minimum(safe_distance[valid] / temperature[valid], 700.0)
    )

    probability = soft / np.maximum(soft.sum(axis=1, keepdims=True), eps)
    mutual = np.sqrt(np.maximum(probability * probability.T, 0.0))

    competition = np.sum(
        probability * (1.0 - probability),
        axis=1,
    )
    scarcity = 1.0 / np.sqrt(np.maximum(competition + 1.0 / degree, eps))
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.55, 2.0)

    endpoint_scarcity = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )
    shared_rank = np.sqrt(np.maximum(rank_score * rank_score.T, 0.0))

    score = inverse
    score *= 0.45 + 0.55 * relative_inverse
    score *= 0.55 + 1.05 * mutual
    score *= 0.70 + 0.50 * shared_rank
    score *= 0.78 + 0.42 * endpoint_scarcity
    score *= 0.90 + 0.10 * soft

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
