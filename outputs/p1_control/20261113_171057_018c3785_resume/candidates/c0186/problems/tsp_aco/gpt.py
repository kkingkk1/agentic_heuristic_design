import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return smooth reciprocal and regret-aware priors for directed TSP edges."""
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

    row_rank = min(1, n - 1)
    col_rank = min(1, n - 1)
    row_second = np.partition(safe, row_rank, axis=1)[:, row_rank]
    col_second = np.partition(safe, col_rank, axis=0)[col_rank, :]

    row_gap = row_second - row_min
    col_gap = col_second - col_min

    row_gap_scale = (
        float(np.median(row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]))
        if np.any(np.isfinite(row_gap) & (row_gap > 0.0))
        else scale
    )
    col_gap_scale = (
        float(np.median(col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]))
        if np.any(np.isfinite(col_gap) & (col_gap > 0.0))
        else scale
    )

    tiny = np.finfo(float).eps
    row_gap_scale = max(row_gap_scale, tiny)
    col_gap_scale = max(col_gap_scale, tiny)

    row_regret = np.divide(
        np.maximum(row_gap, 0.0),
        np.maximum(row_gap, 0.0) + row_gap_scale,
        out=np.ones(n, dtype=float),
        where=np.isfinite(row_gap),
    )
    col_regret = np.divide(
        np.maximum(col_gap, 0.0),
        np.maximum(col_gap, 0.0) + col_gap_scale,
        out=np.ones(n, dtype=float),
        where=np.isfinite(col_gap),
    )

    temperature = max(0.50 * scale, tiny)
    centered = np.where(valid, d - row_min[:, None], 0.0)
    affinity = np.exp(-np.clip(centered / temperature, 0.0, 60.0))
    affinity[~valid] = 0.0

    reverse_valid = valid & valid.T
    pairwise = np.where(
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

    mutual = (
        valid
        & (row_best[:, None] == np.arange(n)[None, :])
        & (col_best[None, :] == np.arange(n)[:, None])
    )

    bonus = (
        0.16 * row_nearest
        + 0.16 * col_nearest
        + 0.22 * mutual.astype(float)
    )

    score = pairwise * (1.0 + bonus)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
