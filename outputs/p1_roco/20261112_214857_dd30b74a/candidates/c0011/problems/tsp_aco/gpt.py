import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    x = d.copy()
    finite_offdiag = np.isfinite(x) & ~np.eye(n, dtype=bool)

    if not np.any(finite_offdiag):
        return np.zeros_like(x, dtype=float)

    positive = x[finite_offdiag & (x > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0

    x[~np.isfinite(x)] = fallback
    x = np.maximum(x, 0.0)
    np.fill_diagonal(x, 0.0)

    positive = x[(~np.eye(n, dtype=bool)) & (x > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0

    endpoint_scale = np.sum(x, axis=1) / max(n - 1, 1)
    endpoint_scale = np.maximum(endpoint_scale, scale * 1e-12)

    local = x / np.sqrt(
        endpoint_scale[:, None] * endpoint_scale[None, :]
    )

    rank_order = np.argsort(
        np.where(np.eye(n, dtype=bool), np.inf, x),
        axis=1,
    )
    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    nearest = np.zeros((n, n), dtype=float)

    for i in range(n):
        nearest[i, rank_order[i, :k]] = 1.0

    mutual = nearest * nearest.T

    row_median = np.median(
        np.where(np.eye(n, dtype=bool), np.nan, x),
        axis=1
    )
    row_median = np.where(
        np.isfinite(row_median) & (row_median > 0),
        row_median,
        scale
    )

    savings = (
        row_median[:, None]
        + row_median[None, :]
        - x
    )
    savings = np.maximum(savings, 0.0)
    savings_scale = np.maximum(
        row_median[:, None] + row_median[None, :],
        scale * 1e-12
    )
    savings = savings / savings_scale

    local_term = np.exp(-np.minimum(local, 50.0))
    score = (
        0.50 * local_term
        + 0.30 * mutual
        + 0.20 * savings
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    offdiag = ~np.eye(n, dtype=bool)
    finite_scores = score[offdiag & np.isfinite(score)]

    if finite_scores.size:
        lo = float(np.min(finite_scores))
        hi = float(np.max(finite_scores))
        if hi > lo:
            score = (score - lo) / (hi - lo)
        else:
            score = np.zeros_like(score)

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)
    return score
