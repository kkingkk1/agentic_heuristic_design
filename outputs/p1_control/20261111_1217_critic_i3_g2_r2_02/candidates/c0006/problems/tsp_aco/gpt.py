import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, distance-adaptive edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_off_diagonal = np.isfinite(distances)
    np.fill_diagonal(finite_off_diagonal, False)
    if not np.any(finite_off_diagonal):
        return np.zeros_like(distances, dtype=float)

    d = np.maximum(distances, 0.0)
    finite_values = d[finite_off_diagonal]
    fallback = float(np.median(finite_values)) if finite_values.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)
    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else fallback
    global_scale = max(global_scale, np.finfo(float).eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest_distances = np.partition(d, k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest_distances, axis=1)
    local_scale = np.maximum(local_scale, global_scale * np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized = d / (pair_scale + global_scale * np.finfo(float).eps)

    distance_attraction = np.exp(-normalized)
    distance_attraction[~np.isfinite(distance_attraction)] = 0.0

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~np.isfinite(rank_score)] = 0.0

    neighborhood = np.zeros((n, n), dtype=float)
    neighborhood[rows, order[:, :k]] = 1.0
    np.fill_diagonal(neighborhood, 0.0)

    mutual = neighborhood * neighborhood.T
    one_sided = np.clip(neighborhood + neighborhood.T - mutual, 0.0, 1.0)

    local_median = np.median(local_scale)
    isolation_ratio = np.maximum(local_scale / max(local_median, global_scale), 1.0)
    bridge_strength = np.sqrt(
        isolation_ratio[:, None] * isolation_ratio[None, :]
    )
    bridge_strength = bridge_strength / (1.0 + bridge_strength)
    bridge_strength = np.clip(bridge_strength, 0.0, 1.0)

    symmetric_rank = (rank_score + rank_score.T) / 2.0
    score = (
        0.55 * distance_attraction
        + 0.20 * symmetric_rank
        + 0.18 * mutual
        + 0.04 * one_sided
        + 0.03 * bridge_strength
    )

    score = (score + score.T) / 2.0
    score[~finite_off_diagonal] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
