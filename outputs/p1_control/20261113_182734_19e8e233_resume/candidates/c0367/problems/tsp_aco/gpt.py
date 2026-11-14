import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank-aware, regret-aware priors for directed TSP edges."""
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
    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)
    row_min = np.where(row_count > 0, row_min, 0.0)
    col_min = np.where(col_count > 0, col_min, 0.0)

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

    row_second = np.full(n, np.inf, dtype=float)
    col_second = np.full(n, np.inf, dtype=float)

    row_mask = row_count >= 2
    col_mask = col_count >= 2

    if np.any(row_mask):
        row_values = safe[row_mask]
        row_second[row_mask] = np.partition(row_values, 1, axis=1)[:, 1]

    if np.any(col_mask):
        col_values = safe[:, col_mask]
        col_second[col_mask] = np.partition(col_values, 1, axis=0)[1, :]

    row_gap = np.where(
        row_mask,
        np.maximum(row_second - row_min, 0.0),
        scale,
    )
    col_gap = np.where(
        col_mask,
        np.maximum(col_second - col_min, 0.0),
        scale,
    )

    row_positive = row_gap[(row_gap > 0.0) & np.isfinite(row_gap)]
    col_positive = col_gap[(col_gap > 0.0) & np.isfinite(col_gap)]

    row_gap_scale = (
        float(np.median(row_positive)) if row_positive.size else scale
    )
    col_gap_scale = (
        float(np.median(col_positive)) if col_positive.size else scale
    )
    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

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
    distance_affinity = np.exp(
        -np.clip(d / temperature, 0.0, 60.0)
    )
    distance_affinity[~valid] = 0.0

    affinity = distance_affinity * (
        0.65 * mutual_rank + 0.35 * regret
    )
    affinity[~valid] = 0.0
    affinity[~np.isfinite(affinity)] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    row_denominator = affinity.sum(axis=1)
    row_scale = np.divide(
        1.0,
        row_denominator,
        out=np.zeros_like(row_denominator),
        where=row_denominator > 0.0,
    )

    col_denominator = affinity.T @ row_scale
    col_scale = np.divide(
        1.0,
        col_denominator,
        out=np.zeros_like(col_denominator),
        where=col_denominator > 0.0,
    )

    for iteration in range(12):
        row_denominator = affinity @ col_scale
        row_scale = np.divide(
            1.0,
            row_denominator,
            out=np.zeros_like(row_denominator),
            where=row_denominator > 0.0,
        )

        col_denominator = affinity.T @ row_scale
        col_scale = np.divide(
            1.0,
            col_denominator,
            out=np.zeros_like(col_denominator),
            where=col_denominator > 0.0,
        )

        if iteration in (3, 7, 11):
            row_error = np.max(
                np.abs(affinity @ col_scale * row_scale - 1.0)
            )
            col_error = np.max(
                np.abs(affinity.T @ row_scale * col_scale - 1.0)
            )
            if max(row_error, col_error) < 1e-7:
                break

    score = affinity * row_scale[:, None] * col_scale[None, :]
    score *= np.power(np.maximum(regret, 0.0) + eps, 0.75)

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
