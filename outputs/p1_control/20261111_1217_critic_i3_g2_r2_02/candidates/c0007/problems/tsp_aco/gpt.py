import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-inclusion priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid_off_diagonal = np.isfinite(distances)
    np.fill_diagonal(valid_off_diagonal, False)

    if not np.any(valid_off_diagonal):
        return np.zeros_like(distances, dtype=float)

    d = np.maximum(distances.copy(), 0.0)

    finite_values = d[valid_off_diagonal]
    fallback = float(np.median(finite_values)) if finite_values.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    d[~np.isfinite(d)] = fallback
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else fallback
    scale = max(scale, np.finfo(float).eps)

    locality = 1.0 / (1.0 + d / scale)
    locality[~np.isfinite(locality)] = 0.0

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~np.isfinite(rank_score)] = 0.0

    neighborhood_size = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    neighborhood = np.zeros((n, n), dtype=float)
    neighborhood[rows, order[:, :neighborhood_size]] = 1.0
    np.fill_diagonal(neighborhood, 0.0)

    mutual = neighborhood * neighborhood.T
    one_sided = np.clip(neighborhood + neighborhood.T - mutual, 0.0, 1.0)

    bilateral_rank = (rank_score + rank_score.T) / 2.0
    score = (
        0.55 * locality
        + 0.25 * bilateral_rank
        + 0.15 * mutual
        + 0.05 * one_sided
    )

    score = (score + score.T) / 2.0
    score[~valid_off_diagonal] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
