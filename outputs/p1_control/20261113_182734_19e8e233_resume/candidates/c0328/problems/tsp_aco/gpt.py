import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank-based local compatibility priors for directed TSP edges."""
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

    finite = np.where(valid, d, np.inf)

    row_count = valid.sum(axis=1).astype(float)
    col_count = valid.sum(axis=0).astype(float)

    safe_row_count = np.maximum(row_count, 1.0)
    safe_col_count = np.maximum(col_count, 1.0)

    row_order = np.argsort(finite, axis=1, kind="stable")
    col_order = np.argsort(finite, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    positions = np.arange(n, dtype=float)
    row_rank[row_order, np.broadcast_to(positions, (n, n))] = (
        positions[:, None]
    )
    col_rank[np.broadcast_to(positions[:, None], (n, n)), col_order] = (
        positions[None, :]
    )

    row_rank = np.where(valid, row_rank, 0.0)
    col_rank = np.where(valid, col_rank, 0.0)

    row_fraction = np.divide(
        row_rank,
        np.maximum(safe_row_count[:, None] - 1.0, 1.0),
    )
    col_fraction = np.divide(
        col_rank,
        np.maximum(safe_col_count[None, :] - 1.0, 1.0),
    )

    row_fraction = np.clip(row_fraction, 0.0, 1.0)
    col_fraction = np.clip(col_fraction, 0.0, 1.0)

    row_locality = np.exp(-4.5 * np.power(row_fraction, 1.35))
    col_locality = np.exp(-4.5 * np.power(col_fraction, 1.35))

    row_top = np.minimum(safe_row_count, 4.0)
    col_top = np.minimum(safe_col_count, 4.0)

    row_core = (row_rank < row_top[:, None]).astype(float)
    col_core = (col_rank < col_top[None, :]).astype(float)

    mutual_core = 1.0 + 0.85 * row_core * col_core
    one_sided_core = 1.0 + 0.20 * np.maximum(row_core, col_core)

    opportunity = np.sqrt(
        np.divide(1.0, safe_row_count)[:, None]
        * np.divide(1.0, safe_col_count)[None, :]
    )
    opportunity /= max(float(np.max(opportunity)), np.finfo(float).eps)
    opportunity = 0.72 + 0.28 * opportunity

    distance_scale = np.zeros(n, dtype=float)
    row_min = np.min(finite, axis=1)
    for i in range(n):
        values = finite[i, valid[i]]
        if values.size:
            positive = values[values > 0.0]
            if positive.size:
                distance_scale[i] = np.median(positive)
            else:
                distance_scale[i] = max(float(np.median(values)), 1.0)

    distance_scale = np.maximum(distance_scale, np.finfo(float).eps)

    relative_distance = np.divide(
        np.where(valid, d, 0.0),
        distance_scale[:, None],
    )
    relative_distance = np.maximum(
        relative_distance - np.divide(row_min, distance_scale),
        0.0,
    )
    distance_term = np.exp(-np.minimum(relative_distance, 50.0))

    score = (
        distance_term
        * np.sqrt(row_locality * col_locality)
        * mutual_core
        * one_sided_core
        * opportunity
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    row_max = np.max(score, axis=1, keepdims=True)
    np.divide(
        score,
        row_max,
        out=result,
        where=row_max > 0.0,
    )

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return result.astype(float, copy=False)
