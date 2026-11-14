import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed TSP edge priors using ordinal costs and two-step lookahead."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    valid_values = d[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = max(float(np.median(valid_values)), 1.0)
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    fallback = scale * float(n + 1)
    outgoing = np.where(np.isfinite(row_best), row_best, fallback)
    incoming = np.where(np.isfinite(col_best), col_best, fallback)

    row_order = np.argsort(masked, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)[None, :]

    col_order = np.argsort(masked, axis=0, kind="stable")
    col_rank = np.empty((n, n), dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n, dtype=float)[:, None]

    row_rank[~valid] = float(n)
    col_rank[~valid] = float(n)

    edge_cost = d / scale
    continuation_cost = (
        outgoing[None, :] + incoming[:, None]
    ) / scale

    total_cost = edge_cost + 0.5 * continuation_cost

    distance_prior = 1.0 / (1.0 + total_cost)
    ordinal_prior = (
        1.0 / (1.0 + row_rank)
        * 1.0 / (1.0 + col_rank)
    )

    score = distance_prior * ordinal_prior
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
