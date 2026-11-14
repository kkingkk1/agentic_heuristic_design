import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")

    n_rows, n_cols = distance_matrix.shape
    if n_rows != n_cols:
        raise ValueError("distance_matrix must be square")

    n = n_rows
    if n == 0:
        return np.empty_like(distance_matrix, dtype=float)
    if n == 1:
        return np.zeros_like(distance_matrix, dtype=float)

    raw = distance_matrix.copy()
    valid = np.isfinite(raw) & (raw >= 0)

    finite_values = raw[valid]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = finite_values[finite_values > 0]
    if positive_values.size:
        fallback = float(np.max(positive_values))
    else:
        fallback = 1.0

    d = raw.copy()
    d[~valid] = fallback * 10.0
    d[d < 0] = fallback * 10.0
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps

    finite_mask = np.isfinite(d)
    row_nearest = np.min(d, axis=1)
    row_nearest = np.where(np.isfinite(row_nearest), row_nearest, fallback)

    masked_for_median = np.where(finite_mask, d, np.nan)
    row_median = np.nanmedian(masked_for_median, axis=1)
    row_median = np.nan_to_num(
        row_median,
        nan=fallback,
        posinf=fallback,
        neginf=fallback,
    )

    local_scale = 0.5 * (row_nearest + row_median)
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = 0.5 * (local_scale[:, None] + local_scale[None, :])
    endpoint_scale = np.maximum(endpoint_scale, eps)

    local_proximity = np.exp(-d / endpoint_scale)
    local_proximity = np.nan_to_num(
        local_proximity,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    ranks = np.argsort(np.argsort(d, axis=1), axis=1)
    reciprocal_rank = 1.0 / (1.0 + ranks.astype(float))
    reciprocal_rank = np.sqrt(
        reciprocal_rank * reciprocal_rank.T
    )

    nearest_rank = reciprocal_rank.copy()
    rank_row_max = np.max(nearest_rank, axis=1, keepdims=True)
    nearest_rank = np.divide(
        nearest_rank,
        np.maximum(rank_row_max, eps),
        out=np.zeros_like(nearest_rank),
        where=rank_row_max > 0,
    )

    row_second = np.partition(d, 1, axis=1)[:, 1]
    savings = (
        row_second[:, None]
        + row_second[None, :]
        - d
    )
    savings = np.maximum(savings, 0.0)

    savings_scale = np.maximum(
        0.5 * (row_second[:, None] + row_second[None, :]),
        eps,
    )
    savings_score = savings / savings_scale
    savings_score = np.nan_to_num(
        savings_score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    row_max = np.max(savings_score, axis=1, keepdims=True)
    savings_score = np.divide(
        savings_score,
        np.maximum(row_max, eps),
        out=np.zeros_like(savings_score),
        where=row_max > 0,
    )

    heuristic = (
        0.45 * local_proximity
        + 0.25 * nearest_rank
        + 0.15 * savings_score
        + 0.15 * reciprocal_rank
    )

    heuristic = 0.5 * (heuristic + heuristic.T)
    heuristic = np.nan_to_num(
        heuristic,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    heuristic = np.maximum(heuristic, 0.0)
    np.fill_diagonal(heuristic, 0.0)

    return heuristic
