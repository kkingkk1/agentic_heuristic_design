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
    fallback = float(np.max(positive_values)) if positive_values.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    d = raw.copy()
    d[~valid] = fallback * 10.0
    d[d < 0] = fallback * 10.0
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    finite_mask = np.isfinite(d)

    row_nearest = np.min(d, axis=1)
    row_nearest = np.where(
        np.isfinite(row_nearest),
        row_nearest,
        fallback,
    )
    row_nearest = np.maximum(row_nearest, eps)

    masked = np.where(finite_mask, d, np.nan)
    row_median = np.nanmedian(masked, axis=1)
    row_median = np.nan_to_num(
        row_median,
        nan=fallback,
        posinf=fallback,
        neginf=fallback,
    )
    row_median = np.maximum(row_median, eps)

    local_scale = np.sqrt(row_nearest * row_median)
    local_scale = np.maximum(local_scale, eps)

    normalized_row_distance = d / local_scale[:, None]
    normalized_col_distance = d / local_scale[None, :]

    exponent = -0.5 * (
        normalized_row_distance + normalized_col_distance
    )
    exponent = np.clip(exponent, -745.0, 0.0)

    endpoint_proximity = np.exp(exponent)
    endpoint_proximity = np.nan_to_num(
        endpoint_proximity,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    ranks = np.argsort(np.argsort(d, axis=1), axis=1)
    rank_score = 1.0 / (1.0 + ranks.astype(float))

    mutual_rank = np.sqrt(rank_score * rank_score.T)
    row_max = np.max(mutual_rank, axis=1, keepdims=True)
    mutual_rank = np.divide(
        mutual_rank,
        np.maximum(row_max, eps),
        out=np.zeros_like(mutual_rank),
        where=row_max > 0,
    )

    heuristic = 0.65 * endpoint_proximity + 0.35 * mutual_rank
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
