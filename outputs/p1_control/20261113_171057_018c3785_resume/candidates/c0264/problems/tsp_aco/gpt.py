import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank- and continuation-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    safe = np.where(valid, d, np.inf)
    positive = d[valid & (d > 0.0)]

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite = d[valid]
        global_scale = float(np.median(finite)) if finite.size else 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    eps = np.finfo(float).eps

    row_counts = valid.sum(axis=1).astype(float)
    col_counts = valid.sum(axis=0).astype(float)

    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.full((n, n), n, dtype=float)
    col_rank = np.full((n, n), n, dtype=float)

    for i in range(n):
        usable = valid[i, row_order[i]]
        positions = np.flatnonzero(usable)
        row_rank[i, row_order[i, positions]] = positions + 1.0

    for j in range(n):
        usable = valid[col_order[:, j], j]
        positions = np.flatnonzero(usable)
        col_rank[col_order[positions, j], j] = positions + 1.0

    row_fraction = np.divide(
        row_rank - 1.0,
        np.maximum(row_counts[:, None] - 1.0, 1.0),
        out=np.ones_like(row_rank),
        where=valid,
    )
    col_fraction = np.divide(
        col_rank - 1.0,
        np.maximum(col_counts[None, :] - 1.0, 1.0),
        out=np.ones_like(col_rank),
        where=valid,
    )

    row_best = np.min(safe, axis=1)
    next_best = np.zeros(n, dtype=float)

    for j in range(n):
        outgoing = safe[j]
        if np.isfinite(outgoing).any():
            next_best[j] = np.min(outgoing[np.isfinite(outgoing)])
        else:
            next_best[j] = global_scale

    continuation = np.zeros_like(d)
    np.subtract(
        next_best[:, None],
        row_best[None, :],
        out=continuation,
        where=np.isfinite(next_best[:, None]) & np.isfinite(row_best[None, :]),
    )
    continuation = np.maximum(continuation, 0.0)

    edge_scale = np.maximum(global_scale, eps)
    distance_term = np.zeros_like(d)
    np.divide(
        d,
        edge_scale + row_best[:, None] + eps,
        out=distance_term,
        where=valid,
    )
    distance_term = np.exp(-np.minimum(distance_term, 40.0))

    continuation_term = np.exp(
        -np.minimum(continuation / edge_scale, 40.0)
    )

    rank_term = np.exp(
        -1.7 * row_fraction - 1.1 * col_fraction
    )

    reverse_valid = valid.T
    reverse_scale = np.zeros_like(d)
    reverse_scale[reverse_valid] = np.exp(
        -np.minimum(
            d.T[reverse_valid] / (edge_scale + d[reverse_valid] + eps),
            20.0,
        )
    )
    reverse_term = np.where(valid, 0.7 + 0.3 * reverse_scale, 0.0)

    score = (
        distance_term
        * continuation_term
        * rank_term
        * reverse_term
    )

    row_totals = np.sum(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_totals,
        out=np.zeros_like(score),
        where=np.isfinite(row_totals) & (row_totals > 0.0),
    )

    score *= np.sqrt(np.maximum(row_fraction, 0.0) * 0.0 + 1.0)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
