import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank-based bidirectional priors for directed TSP edges."""
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

    eps = np.finfo(float).eps
    masked = np.where(valid, d, np.inf)

    row_counts = valid.sum(axis=1).astype(float)
    col_counts = valid.sum(axis=0).astype(float)

    row_order = np.argsort(masked, axis=1, kind="stable")
    col_order = np.argsort(masked, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_positions = np.arange(n, dtype=float)
    col_positions = np.arange(n, dtype=float)

    for i in range(n):
        row_rank[i, row_order[i]] = row_positions
        col_rank[col_order[:, i], i] = col_positions

    row_fraction = np.zeros_like(d, dtype=float)
    col_fraction = np.zeros_like(d, dtype=float)

    np.divide(
        row_rank,
        np.maximum(row_counts[:, None] - 1.0, 1.0),
        out=row_fraction,
        where=valid,
    )
    np.divide(
        col_rank,
        np.maximum(col_counts[None, :] - 1.0, 1.0),
        out=col_fraction,
        where=valid,
    )

    row_median = np.full(n, np.inf, dtype=float)
    col_median = np.full(n, np.inf, dtype=float)

    for i in range(n):
        values = d[i, valid[i]]
        if values.size:
            row_median[i] = np.median(values)

    for j in range(n):
        values = d[valid[:, j], j]
        if values.size:
            col_median[j] = np.median(values)

    row_scale = np.maximum(row_median, eps)
    col_scale = np.maximum(col_median, eps)

    row_relative = np.zeros_like(d, dtype=float)
    col_relative = np.zeros_like(d, dtype=float)

    np.divide(
        d,
        row_scale[:, None],
        out=row_relative,
        where=valid,
    )
    np.divide(
        d,
        col_scale[None, :],
        out=col_relative,
        where=valid,
    )

    row_excess = np.maximum(row_relative - 1.0, 0.0)
    col_excess = np.maximum(col_relative - 1.0, 0.0)

    row_opportunity = 1.0 / np.sqrt(np.maximum(row_counts, 1.0))
    col_opportunity = 1.0 / np.sqrt(np.maximum(col_counts, 1.0))

    opportunity = (
        row_opportunity[:, None] * col_opportunity[None, :]
    )

    rank_score = np.exp(
        -3.2 * np.sqrt(
            np.maximum(row_fraction, 0.0)
            * np.maximum(col_fraction, 0.0)
        )
    )

    local_score = np.exp(
        -0.45 * np.minimum(row_excess + col_excess, 40.0)
    )

    asymmetry_bonus = 1.0 + 0.20 * np.abs(
        row_fraction - col_fraction
    )

    result = rank_score * local_score * opportunity * asymmetry_bonus
    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(result.max())
    if maximum > 0.0:
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result.astype(float, copy=False)
