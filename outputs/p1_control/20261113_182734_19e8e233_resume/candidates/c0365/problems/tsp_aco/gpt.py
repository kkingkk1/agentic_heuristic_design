import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bidirectionally rank-aware priors for directed TSP edges."""
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

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, eps)

    safe = np.where(valid, d, np.inf)

    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    row_min = np.where(np.isfinite(row_min), row_min, 0.0)
    col_min = np.where(np.isfinite(col_min), col_min, 0.0)

    row_delta = np.where(
        valid,
        np.maximum(d - row_min[:, None], 0.0),
        0.0,
    )
    col_delta = np.where(
        valid,
        np.maximum(d - col_min[None, :], 0.0),
        0.0,
    )

    row_gap = np.partition(safe, kth=1, axis=1)[:, 1] - row_min
    col_gap = np.partition(safe, kth=1, axis=0)[1, :] - col_min

    row_gap = np.where(np.isfinite(row_gap) & (row_gap > 0.0), row_gap, scale)
    col_gap = np.where(np.isfinite(col_gap) & (col_gap > 0.0), col_gap, scale)

    row_gap_scale = np.median(row_gap[row_gap > 0.0])
    col_gap_scale = np.median(col_gap[col_gap > 0.0])
    row_gap_scale = max(float(row_gap_scale), eps)
    col_gap_scale = max(float(col_gap_scale), eps)

    regret = np.exp(
        -0.5
        * (
            row_delta / row_gap_scale
            + col_delta / col_gap_scale
        )
    )
    regret[~valid] = 0.0

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[np.argsort(d[i, indices], kind="stable")]
            row_rank[i, order] = np.arange(order.size, dtype=float)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = indices[np.argsort(d[indices, j], kind="stable")]
            col_rank[order, j] = np.arange(order.size, dtype=float)

    row_rank_score = np.divide(
        1.0,
        1.0 + row_rank,
        out=np.zeros_like(row_rank),
        where=valid,
    )
    col_rank_score = np.divide(
        1.0,
        1.0 + col_rank,
        out=np.zeros_like(col_rank),
        where=valid,
    )

    mutual_rank = np.sqrt(
        np.maximum(row_rank_score * col_rank_score, 0.0)
    )

    temperature = max(0.24 * scale, eps)
    distance_logits = np.clip(d / temperature, 0.0, 60.0)
    distance_affinity = np.exp(-distance_logits)
    distance_affinity[~valid] = 0.0

    score = distance_affinity * (
        0.65 * mutual_rank + 0.35 * regret
    )
    score[~valid] = 0.0

    balanced = score
    for _ in range(96):
        previous = balanced.copy()

        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_sum,
            out=np.zeros_like(balanced),
            where=row_sum > 0.0,
        )

        col_sum = balanced.sum(axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            col_sum,
            out=np.zeros_like(balanced),
            where=col_sum > 0.0,
        )

        if np.max(np.abs(balanced - previous)) < 5e-11:
            break

    balanced *= np.power(np.maximum(regret, 0.0) + eps, 0.75)
    balanced[~valid] = 0.0
    balanced[~np.isfinite(balanced)] = 0.0

    maximum = float(np.max(balanced))
    if maximum > 0.0:
        balanced /= maximum

    np.fill_diagonal(balanced, 0.0)
    return balanced.astype(float, copy=False)
