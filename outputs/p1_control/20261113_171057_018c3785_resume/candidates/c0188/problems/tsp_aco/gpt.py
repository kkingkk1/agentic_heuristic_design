import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced reciprocal and regret-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    safe = np.where(valid, d, np.inf)
    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    row_rank = min(1, n - 1)
    col_rank = min(1, n - 1)
    row_second = np.partition(safe, row_rank, axis=1)[:, row_rank]
    col_second = np.partition(safe, col_rank, axis=0)[col_rank, :]

    row_gap = row_second - row_min
    col_gap = col_second - col_min

    row_positive_gaps = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_positive_gaps = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(row_positive_gaps))
        if row_positive_gaps.size
        else scale
    )
    col_gap_scale = (
        float(np.median(col_positive_gaps))
        if col_positive_gaps.size
        else scale
    )

    tiny = np.finfo(float).eps
    row_gap_scale = max(row_gap_scale, tiny)
    col_gap_scale = max(col_gap_scale, tiny)

    row_regret = np.zeros(n, dtype=float)
    col_regret = np.zeros(n, dtype=float)

    finite_row_gap = np.isfinite(row_gap)
    finite_col_gap = np.isfinite(col_gap)

    row_regret[finite_row_gap] = (
        np.maximum(row_gap[finite_row_gap], 0.0)
        / (
            np.maximum(row_gap[finite_row_gap], 0.0)
            + row_gap_scale
        )
    )
    col_regret[finite_col_gap] = (
        np.maximum(col_gap[finite_col_gap], 0.0)
        / (
            np.maximum(col_gap[finite_col_gap], 0.0)
            + col_gap_scale
        )
    )

    temperature = max(0.50 * scale, tiny)
    centered = np.zeros_like(d, dtype=float)
    rows_with_edges = np.isfinite(row_min)
    centered[rows_with_edges] = (
        d[rows_with_edges] - row_min[rows_with_edges, None]
    )

    affinity = np.zeros_like(d, dtype=float)
    affinity[valid] = np.exp(
        -np.clip(centered[valid] / temperature, 0.0, 60.0)
    )

    reverse_valid = valid & valid.T
    score = np.where(
        reverse_valid,
        0.5 * (affinity + affinity.T),
        affinity,
    )

    row_best = np.argmin(safe, axis=1)
    col_best = np.argmin(safe, axis=0)

    row_nearest = np.zeros_like(d, dtype=float)
    col_nearest = np.zeros_like(d, dtype=float)

    for i in range(n):
        j = row_best[i]
        if valid[i, j]:
            row_nearest[i, j] = row_regret[i]

    for j in range(n):
        i = col_best[j]
        if valid[i, j]:
            col_nearest[i, j] = col_regret[j]

    indices = np.arange(n)
    mutual = (
        valid
        & (row_best[:, None] == indices[None, :])
        & (col_best[None, :] == indices[:, None])
    )

    bonus = (
        0.16 * row_nearest
        + 0.16 * col_nearest
        + 0.22 * mutual.astype(float)
    )

    score *= 1.0 + bonus
    score[~valid] = 0.0

    for _ in range(12):
        row_max = np.max(score, axis=1, keepdims=True)
        row_scale = np.where(row_max > 0.0, row_max, 1.0)
        score = np.divide(score, row_scale)

        col_max = np.max(score, axis=0, keepdims=True)
        col_scale = np.where(col_max > 0.0, col_max, 1.0)
        score = np.divide(score, col_scale)

        score[~valid] = 0.0

    score = np.power(np.maximum(score, 0.0), 1.15)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
