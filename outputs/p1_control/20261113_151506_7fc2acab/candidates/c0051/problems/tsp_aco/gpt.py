import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]

    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distance_matrix.copy()
    d[~np.isfinite(d)] = np.nan
    d = np.maximum(d, 0.0)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    global_fallback = float(np.median(finite_values))
    d = np.where(np.isfinite(d), d, global_fallback)

    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    off_diagonal = d[np.isfinite(d)]
    if off_diagonal.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = off_diagonal[off_diagonal > 0.0]
    if positive_values.size:
        distance_scale = float(np.median(positive_values))
    else:
        distance_scale = 1.0

    eps = np.finfo(float).eps
    distance_scale = max(distance_scale, eps)

    finite_max = float(np.max(off_diagonal))
    d[~np.isfinite(d)] = finite_max

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks = ranks / max(float(n - 1), 1.0)

    nearest_k = max(1, min(4, n - 1))
    nearest = np.zeros((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    nearest[rows, order[:, :nearest_k]] = 1.0

    mutual = nearest * nearest.T
    one_sided = 0.5 * (nearest + nearest.T)

    local_k = max(1, min(3, n - 1))
    local_scale = np.partition(d, local_k - 1, axis=1)[:, local_k - 1]
    local_scale = np.maximum(local_scale, distance_scale * 1e-6)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :] + distance_scale**2 * 1e-12
    )

    relative_distance = d / pair_scale
    local_affinity = np.exp(-np.maximum(relative_distance, 0.0))

    robust_distance = d / distance_scale
    inverse_affinity = np.exp(-np.minimum(robust_distance, 50.0))

    rank_affinity = np.exp(
        -0.5 * (ranks + ranks.T)
    )

    long_edge_penalty = np.exp(
        -0.5 * np.maximum(relative_distance - 1.0, 0.0) ** 2
    )

    score = (
        0.38 * inverse_affinity
        + 0.24 * local_affinity
        + 0.16 * rank_affinity
        + 0.10 * one_sided
        + 0.08 * mutual
    )

    score *= 0.75 + 0.25 * long_edge_penalty
    np.fill_diagonal(score, 0.0)

    score_min = float(np.min(score))
    score_max = float(np.max(score))

    if score_max > score_min:
        score = (score - score_min) / (score_max - score_min)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return np.asarray(score, dtype=float)
