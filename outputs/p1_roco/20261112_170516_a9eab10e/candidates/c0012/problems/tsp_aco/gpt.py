import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.where(np.isfinite(d), np.maximum(d, 0.0), np.inf)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_edges = d[np.isfinite(d)]
    if finite_edges.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_edges = finite_edges[finite_edges > 0]
    if positive_edges.size:
        scale = float(np.median(positive_edges))
    else:
        scale = 1.0
    scale = max(scale, np.finfo(float).eps)

    order = np.argsort(d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    nearest = order[:, 0]
    mutual = (nearest[:, None] == np.arange(n)[None, :]) & (
        nearest[None, :] == np.arange(n)[:, None]
    )

    local_scale = np.partition(d, min(2, n - 1), axis=1)[:, min(2, n - 1)]
    local_scale = np.where(np.isfinite(local_scale), local_scale, scale)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    distance_term = np.exp(-d / scale)
    rank_term = 1.0 / np.sqrt((ranks + 1.0) * (ranks.T + 1.0))
    locality_term = np.sqrt(
        (local_scale[:, None] * local_scale[None, :])
    ) / np.maximum(d, np.finfo(float).eps)
    locality_term = np.clip(locality_term, 0.0, 4.0)

    score = distance_term * (0.55 + 0.45 * rank_term)
    score *= 0.75 + 0.25 * locality_term
    score *= np.where(mutual, 1.8, 1.0)

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0,
    )

    return score
