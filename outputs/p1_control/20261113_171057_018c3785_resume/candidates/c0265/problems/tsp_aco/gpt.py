import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank-based candidate priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    safe = np.where(valid, d, np.inf)

    row_counts = valid.sum(axis=1)
    col_counts = valid.sum(axis=0)

    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.full((n, n), n, dtype=float)
    col_rank = np.full((n, n), n, dtype=float)

    row_positions = np.broadcast_to(
        np.arange(n, dtype=float),
        (n, n),
    )
    col_positions = row_positions.T

    np.put_along_axis(row_rank, row_order, row_positions, axis=1)
    np.put_along_axis(col_rank, col_order, col_positions, axis=0)

    row_denominator = np.maximum(row_counts[:, None] - 1, 1)
    col_denominator = np.maximum(col_counts[None, :] - 1, 1)

    row_quality = 1.0 - row_rank / row_denominator
    col_quality = 1.0 - col_rank / col_denominator

    row_quality = np.clip(row_quality, 0.0, 1.0)
    col_quality = np.clip(col_quality, 0.0, 1.0)

    valid_values = d[valid]
    global_scale = float(np.median(valid_values))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    row_span = np.full(n, global_scale, dtype=float)
    col_span = np.full(n, global_scale, dtype=float)

    for i in range(n):
        values = d[i, valid[i]]
        if values.size > 1:
            span = float(np.percentile(values, 75) - np.percentile(values, 25))
            if np.isfinite(span) and span > eps:
                row_span[i] = span

    for j in range(n):
        values = d[valid[:, j], j]
        if values.size > 1:
            span = float(np.percentile(values, 75) - np.percentile(values, 25))
            if np.isfinite(span) and span > eps:
                col_span[j] = span

    row_cost = np.zeros_like(d)
    col_cost = np.zeros_like(d)

    np.divide(
        np.maximum(d - row_min[:, None], 0.0),
        row_span[:, None],
        out=row_cost,
        where=valid,
    )
    np.divide(
        np.maximum(d - col_min[None, :], 0.0),
        col_span[None, :],
        out=col_cost,
        where=valid,
    )

    local_cost = np.exp(
        -np.minimum(row_cost + col_cost, 60.0)
    )

    candidate_width = max(2, int(np.ceil(np.sqrt(n))))
    candidate_width = min(candidate_width, max(n - 1, 1))

    row_candidate = row_rank < candidate_width
    col_candidate = col_rank < candidate_width

    support = (
        1.0
        + 1.5 * row_candidate.astype(float)
        + 1.5 * col_candidate.astype(float)
        + 2.5 * (row_candidate & col_candidate).astype(float)
    )

    rank_prior = (
        np.maximum(row_quality, 0.0)
        * np.maximum(col_quality, 0.0)
    )

    score = local_cost * support * (
        0.35 + 0.65 * rank_prior
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
