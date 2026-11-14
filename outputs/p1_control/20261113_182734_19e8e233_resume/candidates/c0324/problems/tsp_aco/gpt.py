import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank-based, bridge-aware priors for directed TSP edges."""
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
    safe = np.where(valid, d, np.inf)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, eps)

    row_best = np.min(safe, axis=1)
    row_best = np.where(row_count > 0, row_best, 0.0)

    col_best = np.min(safe, axis=0)
    col_best = np.where(col_count > 0, col_best, 0.0)

    row_rank = np.zeros_like(d, dtype=float)
    col_rank = np.zeros_like(d, dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[np.argsort(d[i, indices], kind="mergesort")]
            row_rank[i, order] = np.arange(1.0, indices.size + 1.0)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = indices[np.argsort(d[indices, j], kind="mergesort")]
            col_rank[order, j] = np.arange(1.0, indices.size + 1.0)

    row_rank_score = np.zeros_like(d, dtype=float)
    col_rank_score = np.zeros_like(d, dtype=float)

    np.divide(
        1.0,
        row_rank,
        out=row_rank_score,
        where=valid,
    )
    np.divide(
        1.0,
        col_rank,
        out=col_rank_score,
        where=valid,
    )

    relative_distance = np.zeros_like(d, dtype=float)
    np.divide(
        d,
        scale + np.abs(row_best[:, None]),
        out=relative_distance,
        where=valid,
    )
    inverse_distance = np.zeros_like(d, dtype=float)
    np.divide(
        1.0,
        1.0 + relative_distance,
        out=inverse_distance,
        where=valid,
    )

    outgoing_quality = np.zeros(n, dtype=float)
    for j in range(n):
        candidates = np.flatnonzero(valid[j])
        if candidates.size:
            continuation = d[j, candidates]
            continuation = continuation[np.isfinite(continuation)]
            if continuation.size:
                continuation_scale = max(float(np.median(continuation)), eps)
                outgoing_quality[j] = float(
                    np.mean(1.0 / (1.0 + continuation / continuation_scale))
                )

    incoming_pressure = np.zeros(n, dtype=float)
    max_col_count = max(int(np.max(col_count)), 1)
    incoming_pressure = col_count.astype(float) / float(max_col_count)

    bridge = outgoing_quality[None, :] * (1.0 - 0.35 * incoming_pressure[None, :])
    bridge = np.broadcast_to(bridge, d.shape)

    sparsity = np.zeros_like(d, dtype=float)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            keep = max(2, int(np.ceil(np.sqrt(indices.size))))
            keep = min(keep, indices.size)
            ordered = indices[np.argsort(d[i, indices], kind="mergesort")]
            sparsity[i, ordered[:keep]] = 1.0

    score = (
        0.45 * row_rank_score
        + 0.25 * col_rank_score
        + 0.20 * inverse_distance
        + 0.10 * bridge
    )

    score *= 0.55 + 0.45 * sparsity
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
