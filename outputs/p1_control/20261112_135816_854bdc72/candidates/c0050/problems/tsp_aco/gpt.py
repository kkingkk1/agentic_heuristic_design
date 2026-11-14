import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric nonnegative edge-prior matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d >= 0) & ~diagonal

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0]
    eps = np.finfo(float).eps

    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.median(finite_values)), 1.0)
    )
    global_scale = max(global_scale, eps)

    safe = np.where(valid, d, np.inf)
    safe[diagonal] = np.inf

    # Local scale from the closest neighborhood of each endpoint.
    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    ordered = np.sort(safe, axis=1)
    local_scale = ordered[:, :k]
    local_scale = np.where(np.isfinite(local_scale), local_scale, global_scale)
    local_scale = np.median(local_scale, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, eps)

    normalized_distance = safe / pair_scale
    normalized_distance = np.where(np.isfinite(normalized_distance), normalized_distance, 60.0)

    # Shortness relative to both endpoint neighborhoods.
    closeness = np.exp(-np.minimum(normalized_distance, 60.0))

    # Soft rank preference from each endpoint.
    ranks = np.full((n, n), n, dtype=float)
    for i in range(n):
        order = np.argsort(safe[i], kind="stable")
        order = order[np.isfinite(safe[i, order])]
        ranks[i, order] = np.arange(order.size, dtype=float)

    rank_temperature = max(1.0, np.sqrt(n))
    rank_preference = np.exp(-ranks / rank_temperature)
    mutual_rank = np.sqrt(rank_preference * rank_preference.T)

    # Mutual k-nearest-neighbor support.
    nearest_mask = np.zeros((n, n), dtype=float)
    nearest_k = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    for i in range(n):
        order = np.argsort(safe[i], kind="stable")
        order = order[np.isfinite(safe[i, order])]
        nearest_mask[i, order[:nearest_k]] = 1.0

    mutual_support = nearest_mask * nearest_mask.T
    one_sided_support = 0.5 * (nearest_mask + nearest_mask.T)
    neighborhood_support = 1.0 + 0.35 * one_sided_support + 0.65 * mutual_support

    # Penalize edges incident to nodes with many similarly attractive alternatives.
    competition = np.sum(np.exp(-np.minimum(normalized_distance, 20.0)), axis=1)
    competition = np.maximum(competition, eps)
    diversity = 1.0 / np.sqrt(competition[:, None] * competition[None, :])

    # Geometric combination rewards consistent endpoint-level evidence.
    score = (
        np.maximum(closeness, eps) ** 0.45
        * np.maximum(mutual_rank, eps) ** 0.30
        * np.maximum(diversity, eps) ** 0.15
        * np.maximum(neighborhood_support, eps) ** 0.10
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
