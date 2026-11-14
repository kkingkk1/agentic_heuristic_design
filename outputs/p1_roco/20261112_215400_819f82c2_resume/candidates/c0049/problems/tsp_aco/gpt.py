import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-prior indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square two-dimensional array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distances.copy()
    diagonal = np.eye(n, dtype=bool)

    finite_offdiag = np.isfinite(d) & ~diagonal
    positive = finite_offdiag & (d > 0)

    if not np.any(positive):
        return np.zeros((n, n), dtype=float)

    valid_values = d[positive]
    fallback = float(np.median(valid_values))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    invalid = ~np.isfinite(d) | (d < 0)
    d[invalid] = fallback
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    safe_d = np.maximum(d, eps)

    k = min(n - 1, max(2, int(np.sqrt(n))))
    ordered = np.sort(safe_d, axis=1)
    local_scale = np.median(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        np.outer(local_scale, local_scale)
    )
    relative_shortness = np.exp(-safe_d / (pair_scale + eps))

    reciprocal_rank = np.zeros((n, n), dtype=float)
    neighborhood = np.zeros((n, n), dtype=bool)

    for i in range(n):
        order = np.argsort(safe_d[i])
        order = order[order != i]
        selected = order[:k]
        neighborhood[i, selected] = True

        for rank, j in enumerate(order):
            reciprocal_rank[i, j] = 1.0 / (rank + 1.0)

    rank_score = np.sqrt(reciprocal_rank * reciprocal_rank.T)

    mutual_bonus = (
        neighborhood & neighborhood.T
    ).astype(float)

    strength = np.exp(-safe_d / (local_scale[:, None] + eps))
    strength[diagonal] = 0.0
    endpoint_pressure = 1.0 / (
        np.sum(strength, axis=1, keepdims=True) + eps
    )
    scarcity = np.sqrt(
        endpoint_pressure * endpoint_pressure.T
    )
    scarcity /= np.max(scarcity) + eps

    inverse_distance = 1.0 / safe_d
    inverse_distance[diagonal] = 0.0
    inverse_distance /= np.max(inverse_distance) + eps

    score = (
        0.40 * relative_shortness
        + 0.25 * rank_score
        + 0.20 * inverse_distance
        + 0.10 * mutual_bonus
        + 0.05 * scarcity
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
