import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank-based, two-step lookahead priors for directed TSP edges."""
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

    tiny = np.finfo(float).eps
    finite_edges = d[valid]
    positive_edges = finite_edges[finite_edges > 0.0]

    if positive_edges.size:
        scale = float(np.median(positive_edges))
    else:
        scale = float(np.median(finite_edges))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, tiny)

    masked = np.where(valid, d, np.inf)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_has_edge = np.any(valid, axis=1)
    col_has_edge = np.any(valid, axis=0)

    row_best = np.where(row_has_edge, row_best, scale)
    col_best = np.where(col_has_edge, col_best, scale)

    # Rank each outgoing edge within its source row.
    row_order = np.argsort(masked, axis=1, kind="stable")
    row_rank_positions = np.argsort(row_order, axis=1, kind="stable") + 1
    row_rank = row_rank_positions.astype(float)
    row_rank[~valid] = np.inf

    # Rank each incoming edge within its destination column.
    col_order = np.argsort(masked, axis=0, kind="stable")
    col_rank_positions = np.argsort(col_order, axis=0, kind="stable") + 1
    col_rank = col_rank_positions.astype(float)
    col_rank[~valid] = np.inf

    # Best continuation after reaching each destination.
    next_out = np.min(masked, axis=1)
    next_out = np.where(row_has_edge, next_out, scale)

    # Best predecessor before leaving each source.
    next_in = np.min(masked, axis=0)
    next_in = np.where(col_has_edge, next_in, scale)

    lookahead = np.zeros_like(d, dtype=float)
    lookahead[valid] = (
        d[valid]
        + next_out[np.nonzero(valid)[1]]
        + next_in[np.nonzero(valid)[0]]
    )

    relative_row_cost = np.zeros_like(d, dtype=float)
    relative_col_cost = np.zeros_like(d, dtype=float)

    np.divide(
        np.maximum(d - row_best[:, None], 0.0),
        scale,
        out=relative_row_cost,
        where=valid,
    )
    np.divide(
        np.maximum(d - col_best[None, :], 0.0),
        scale,
        out=relative_col_cost,
        where=valid,
    )

    local_quality = 1.0 / (
        1.0
        + relative_row_cost
        + relative_col_cost
    )

    lookahead_quality = np.zeros_like(d, dtype=float)
    lookahead_quality[valid] = 1.0 / (
        1.0 + lookahead[valid] / scale
    )

    rank_quality = np.zeros_like(d, dtype=float)
    rank_quality[valid] = 1.0 / np.sqrt(
        row_rank[valid] * col_rank[valid]
    )

    # Favor edges whose endpoints both have few similarly competitive alternatives.
    row_margin = np.zeros(n, dtype=float)
    col_margin = np.zeros(n, dtype=float)

    for i in range(n):
        candidates = masked[i, valid[i]]
        if candidates.size >= 2:
            ordered = np.sort(candidates)
            row_margin[i] = ordered[1] - ordered[0]
        elif candidates.size == 1:
            row_margin[i] = scale

    for j in range(n):
        candidates = masked[valid[:, j], j]
        if candidates.size >= 2:
            ordered = np.sort(candidates)
            col_margin[j] = ordered[1] - ordered[0]
        elif candidates.size == 1:
            col_margin[j] = scale

    scarcity = np.zeros_like(d, dtype=float)
    scarcity[valid] = 1.0 / (
        1.0
        + (row_margin[np.nonzero(valid)[0]]
           + col_margin[np.nonzero(valid)[1]]) / scale
    )

    score = (
        local_quality
        * lookahead_quality
        * rank_quality
        * (0.5 + 0.5 * scarcity)
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
