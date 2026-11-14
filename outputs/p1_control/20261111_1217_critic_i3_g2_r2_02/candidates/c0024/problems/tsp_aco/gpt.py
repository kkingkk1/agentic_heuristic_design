import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-inclusion priorities for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distances.copy()
    valid = np.isfinite(d)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    finite_values = d[valid]
    fallback = float(np.max(finite_values))
    d[~np.isfinite(d)] = fallback
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    ranks[~np.isfinite(ranks)] = n

    local_k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    local_neighbors = np.take_along_axis(d, order[:, :local_k], axis=1)
    local_scale = np.median(local_neighbors, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    endpoint_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    endpoint_scale = np.maximum(endpoint_scale, np.finfo(float).eps)

    normalized_distance = d / endpoint_scale

    # Two-sided endpoint-relative attraction.
    local_affinity = 1.0 / (1.0 + normalized_distance)
    local_affinity[~np.isfinite(local_affinity)] = 0.0

    # Smooth preference for edges that are highly ranked from both endpoints.
    rank_affinity = 1.0 / (1.0 + np.minimum(ranks, ranks.T))
    rank_affinity[~np.isfinite(rank_affinity)] = 0.0

    # Mutual neighborhood compatibility, kept moderate to avoid over-concentration.
    nearest = np.zeros((n, n), dtype=float)
    nearest[rows, order[:, :local_k]] = 1.0
    np.fill_diagonal(nearest, 0.0)

    mutual = nearest * nearest.T
    one_sided = np.maximum(nearest, nearest.T)

    # Bridge preservation: locally isolated endpoints retain some probability
    # of selecting edges beyond their nearest-neighbor basin.
    isolation = local_scale / (global_scale + local_scale)
    isolation = np.clip(isolation, 0.0, 1.0)
    bridge_bonus = 0.5 * (isolation[:, None] + isolation[None, :])

    # A small distance floor prevents long but potentially necessary bridges
    # from receiving exactly zero prior.
    distance_floor = 1.0 / (1.0 + d / global_scale)
    distance_floor[~np.isfinite(distance_floor)] = 0.0

    score = (
        0.46 * local_affinity
        + 0.24 * rank_affinity
        + 0.12 * mutual
        + 0.06 * one_sided
        + 0.08 * bridge_bonus
        + 0.04 * distance_floor
    )

    score = (score + score.T) / 2.0
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
