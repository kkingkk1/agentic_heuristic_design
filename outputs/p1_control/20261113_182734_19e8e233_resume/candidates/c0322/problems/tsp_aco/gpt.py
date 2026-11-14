import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return sparse rank-based priors with reciprocal and look-ahead signals."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    result = np.zeros((n, n), dtype=float)
    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    masked = np.where(valid, d, np.inf)

    row_order = np.argsort(masked, axis=1, kind="stable")
    col_order = np.argsort(masked, axis=0, kind="stable")

    row_rank = np.full((n, n), n + 1, dtype=float)
    col_rank = np.full((n, n), n + 1, dtype=float)

    row_positions = np.arange(n, dtype=float)[None, :]
    col_positions = np.arange(n, dtype=float)[:, None]

    row_rank[row_order] = np.broadcast_to(
        row_positions,
        (n, n),
    )
    col_rank[col_order] = np.broadcast_to(
        col_positions,
        (n, n),
    )

    row_rank += 1.0
    col_rank += 1.0
    row_rank[~valid] = n + 1.0
    col_rank[~valid] = n + 1.0

    valid_counts = valid.sum(axis=1)
    row_best = np.min(masked, axis=1)
    row_best = np.where(valid_counts > 0, row_best, 0.0)

    positive = d[valid & (d > 0.0)]
    scale = (
        float(np.median(positive))
        if positive.size
        else max(float(np.median(d[valid])), 1.0)
    )
    scale = max(scale, eps)

    continuation = row_best[np.newaxis, :].T
    continuation = np.broadcast_to(continuation, (n, n))
    lookahead = d + 0.7 * continuation

    finite_lookahead = lookahead[valid & np.isfinite(lookahead)]
    lookahead_scale = (
        float(np.median(finite_lookahead))
        if finite_lookahead.size
        else scale
    )
    lookahead_scale = max(lookahead_scale, eps)

    k = max(2, min(n - 1, int(np.ceil(np.sqrt(n)))))
    candidate = valid & (
        (row_rank <= k) | (col_rank <= k)
    )

    reverse_valid = valid.T
    reverse_distance = d.T
    reciprocal = np.zeros((n, n), dtype=float)
    reciprocal[valid & reverse_valid] = 1.0 / (
        1.0 + np.abs(d[valid & reverse_valid] - reverse_distance[valid & reverse_valid])
        / (scale + eps)
    )

    row_term = 1.0 / np.power(row_rank, 1.35)
    col_term = 1.0 / np.power(col_rank, 0.85)
    continuation_term = 1.0 / np.power(
        1.0 + np.maximum(lookahead, 0.0) / lookahead_scale,
        1.4,
    )

    score = row_term * col_term * continuation_term
    score *= 1.0 + 0.45 * reciprocal
    score[~candidate] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
