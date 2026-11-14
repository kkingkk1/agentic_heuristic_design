import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, diversity-preserving priors for directed TSP edges."""
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

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe = np.where(valid, d, np.inf)
    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    row_min_safe = np.where(np.isfinite(row_min), row_min, 0.0)
    col_min_safe = np.where(np.isfinite(col_min), col_min, 0.0)

    centered_row = np.maximum(d - row_min_safe[:, None], 0.0)
    centered_col = np.maximum(d - col_min_safe[None, :], 0.0)
    centered_row[~valid] = 0.0
    centered_col[~valid] = 0.0

    row_rank = np.full((n, n), n, dtype=int)
    col_rank = np.full((n, n), n, dtype=int)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[np.argsort(d[i, indices], kind="stable")]
            row_rank[i, order] = np.arange(order.size)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = indices[np.argsort(d[indices, j], kind="stable")]
            col_rank[order, j] = np.arange(order.size)

    row_best = row_rank == 0
    col_best = col_rank == 0
    mutual = row_best & col_best

    row_second = np.full(n, np.inf, dtype=float)
    col_second = np.full(n, np.inf, dtype=float)

    for i in range(n):
        values = d[i, valid[i]]
        if values.size >= 2:
            row_second[i] = np.partition(values, 1)[1]

    for j in range(n):
        values = d[valid[:, j], j]
        if values.size >= 2:
            col_second[j] = np.partition(values, 1)[1]

    row_gap = np.maximum(
        np.where(np.isfinite(row_second), row_second, row_min_safe) - row_min_safe,
        0.0,
    )
    col_gap = np.maximum(
        np.where(np.isfinite(col_second), col_second, col_min_safe) - col_min_safe,
        0.0,
    )

    positive_row_gap = row_gap[row_gap > 0.0]
    positive_col_gap = col_gap[col_gap > 0.0]

    row_gap_scale = (
        float(np.median(positive_row_gap))
        if positive_row_gap.size
        else scale
    )
    col_gap_scale = (
        float(np.median(positive_col_gap))
        if positive_col_gap.size
        else scale
    )

    row_gap_scale = max(row_gap_scale, np.finfo(float).eps)
    col_gap_scale = max(col_gap_scale, np.finfo(float).eps)

    row_regret = np.exp(
        -np.clip(centered_row / row_gap_scale, 0.0, 60.0)
    )
    col_regret = np.exp(
        -np.clip(centered_col / col_gap_scale, 0.0, 60.0)
    )
    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    temperature = max(0.24 * scale, np.finfo(float).eps)
    affinity = np.exp(-np.clip(centered_row / temperature, 0.0, 60.0))
    affinity[~valid] = 0.0

    row_counts = valid.sum(axis=1, keepdims=True).astype(float)
    uniform_floor = np.divide(
        valid.astype(float),
        row_counts,
        out=np.zeros_like(affinity),
        where=row_counts > 0.0,
    )

    row_peak = np.max(affinity, axis=1, keepdims=True)
    row_entropy_proxy = np.divide(
        affinity.sum(axis=1, keepdims=True) - row_peak,
        np.maximum(affinity.sum(axis=1, keepdims=True), 1.0),
    )
    floor_weight = 0.025 + 0.10 * np.clip(1.0 - row_entropy_proxy, 0.0, 1.0)

    prior = (1.0 - floor_weight) * affinity + floor_weight * uniform_floor
    prior[~valid] = 0.0

    balanced = prior.copy()
    for _ in range(64):
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

    competitive = (row_rank <= 1) | (col_rank <= 1)
    selective_regret = np.where(competitive, regret, 1.0)

    score = balanced * np.power(selective_regret, 0.45)
    score *= 1.0 + 0.18 * mutual.astype(float)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
