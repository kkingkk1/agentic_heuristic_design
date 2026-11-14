import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, nonnegative edge-priority matrix for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must be nonnegative")

    n = distances.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    d = distances.copy()
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    k = max(1, min(n - 1, int(np.sqrt(n))))
    sorted_distances = np.sort(d, axis=1)

    local_scale = np.median(sorted_distances[:, :k], axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    normalized_distance = d / global_scale
    inverse_distance = 1.0 / (normalized_distance + 1e-12)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    row_min = np.min(d, axis=1)
    row_min = np.where(np.isfinite(row_min), row_min, global_scale)

    regret = np.maximum(d - row_min[:, None], 0.0)
    competition = 1.0 / (1.0 + regret / global_scale)
    competition[~np.isfinite(competition)] = 0.0

    local_compatibility = np.exp(
        -d / (local_scale[:, None] + local_scale[None, :] + 1e-12)
    )
    local_compatibility[~np.isfinite(local_compatibility)] = 0.0

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(1, n + 1, dtype=float)
    ranks = np.minimum(ranks, float(n - 1))
    np.fill_diagonal(ranks, np.inf)

    mutual_rank = 1.0 / (ranks + ranks.T)
    mutual_rank[~np.isfinite(mutual_rank)] = 0.0

    nearest = np.argmin(d, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j != i and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    score = (
        inverse_distance
        * (0.55 + 0.45 * np.sqrt(competition * competition.T))
        * (0.65 + 0.35 * local_compatibility)
        * (1.0 + 0.20 * mutual_rank)
        * (1.0 + 0.25 * mutual_nearest)
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
