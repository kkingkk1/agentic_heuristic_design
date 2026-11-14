import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(distances)

    valid_values = distances[finite & off_diagonal]
    if valid_values.size == 0:
        return np.zeros_like(distances, dtype=float)

    nonnegative_values = np.maximum(valid_values, 0.0)
    positive_values = nonnegative_values[nonnegative_values > 0]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    valid = finite & off_diagonal
    safe = np.maximum(distances, 0.0)

    # Symmetrize finite edges and preserve invalid edges as unavailable.
    symmetric = 0.5 * (safe + safe.T)
    symmetric_valid = valid & valid.T
    d = np.where(symmetric_valid, symmetric, np.inf)
    np.fill_diagonal(d, np.inf)

    available = np.isfinite(d)
    if not np.any(available):
        return np.zeros_like(distances, dtype=float)

    positive = d[available & (d > 0)]
    scale = float(np.median(positive)) if positive.size else global_scale
    scale = max(scale, np.finfo(float).eps)

    eps = np.finfo(float).eps * scale
    inverse = 1.0 / (d + eps)
    inverse[~available] = 0.0
    inverse /= max(float(np.max(inverse)), np.finfo(float).eps)

    # Robust local scale for each endpoint.
    local_scale = np.full(n, scale, dtype=float)
    for i in range(n):
        row_positive = d[i, np.isfinite(d[i]) & (d[i] > 0)]
        if row_positive.size:
            local_scale[i] = max(
                float(np.median(row_positive)),
                np.finfo(float).eps,
            )

    normalized_distance = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = np.where(
        np.isfinite(normalized_distance),
        normalized_distance,
        np.inf,
    )

    local_attractiveness = np.exp(
        -np.minimum(normalized_distance, 60.0)
    )
    local_attractiveness[~available] = 0.0

    # Soft rank preference for each endpoint.
    order = np.argsort(
        np.where(available, d, np.inf),
        axis=1,
        kind="stable",
    )
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = np.exp(-ranks / max(1.0, np.sqrt(float(n))))
    rank_score[~available] = 0.0

    # Mutual-neighborhood support.
    nearest_k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    neighborhood = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = order[i, np.isfinite(d[i, order])]
        neighborhood[i, candidates[:nearest_k]] = 1.0

    mutual_support = neighborhood * neighborhood.T
    endpoint_support = np.sqrt(
        np.maximum(rank_score, 0.0) * np.maximum(rank_score.T, 0.0)
    )

    # Mild diversity correction for edges incident to crowded nodes.
    competition = np.sum(local_attractiveness, axis=1)
    competition = np.maximum(competition, np.finfo(float).eps)
    diversity = 1.0 / np.sqrt(
        competition[:, None] * competition[None, :]
    )

    score = (
        np.maximum(inverse, 1e-12) ** 0.40
        * np.maximum(local_attractiveness, 1e-12) ** 0.30
        * np.maximum(0.5 + 0.5 * endpoint_support, 1e-12) ** 0.15
        * np.maximum(1.0 + 0.5 * mutual_support, 1e-12) ** 0.10
        * np.maximum(diversity, 1e-12) ** 0.05
    )

    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    valid_scores = score[symmetric_valid]
    mean_score = float(np.mean(valid_scores)) if valid_scores.size else 0.0
    if mean_score > 0.0:
        score /= mean_score

    return score.astype(float, copy=False)
