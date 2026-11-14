import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic rank-consensus priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    eps = np.finfo(float).eps
    masked = np.where(valid, d, np.inf)

    row_order = np.argsort(masked, axis=1, kind="stable")
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)

    col_order = np.argsort(masked, axis=0, kind="stable")
    col_rank = np.empty_like(col_order, dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n, dtype=float)

    finite_values = d[valid]
    global_order = np.argsort(finite_values, kind="stable")
    global_rank = np.empty_like(global_order, dtype=float)
    global_rank[global_order] = np.arange(global_order.size, dtype=float)

    global_percentile = np.zeros_like(d, dtype=float)
    if finite_values.size > 1:
        global_percentile[valid] = (
            global_rank / float(finite_values.size - 1)
        )
    else:
        global_percentile[valid] = 0.0

    row_median = np.full(n, 0.0, dtype=float)
    col_median = np.full(n, 0.0, dtype=float)

    for i in range(n):
        values = d[i, valid[i]]
        if values.size:
            row_median[i] = float(np.median(values))

    for j in range(n):
        values = d[valid[:, j], j]
        if values.size:
            col_median[j] = float(np.median(values))

    row_scale = np.maximum(row_median, eps)
    col_scale = np.maximum(col_median, eps)

    local_scale = (
        2.0 * row_scale[:, None] * col_scale[None, :]
        / (row_scale[:, None] + col_scale[None, :])
    )

    relative_length = np.zeros_like(d, dtype=float)
    np.divide(
        d,
        local_scale,
        out=relative_length,
        where=valid,
    )

    rank_support = (
        1.0 / np.power(1.0 + row_rank, 1.35)
        * 1.0 / np.power(1.0 + col_rank, 1.05)
    )

    length_support = np.exp(
        -np.minimum(relative_length, 50.0) ** 0.85
    )

    percentile_support = np.power(
        np.maximum(1.0 - global_percentile, 0.0),
        1.25,
    )

    score = (
        rank_support
        * length_support
        * (0.35 + 0.65 * percentile_support)
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
