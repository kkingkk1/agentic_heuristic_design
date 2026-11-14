import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")

    n_rows, n_cols = distance_matrix.shape
    if n_rows != n_cols:
        raise ValueError("distance_matrix must be square")

    n = n_rows
    heuristic = np.zeros((n, n), dtype=float)

    if n <= 1:
        return heuristic

    finite_nonnegative = np.isfinite(distance_matrix) & (distance_matrix >= 0)
    positive_values = distance_matrix[
        finite_nonnegative & (distance_matrix > 0)
    ]

    fallback = (
        float(np.max(positive_values))
        if positive_values.size
        else 1.0
    )
    fallback = max(fallback, np.finfo(float).eps)

    d = np.asarray(distance_matrix, dtype=float).copy()
    d[~finite_nonnegative] = fallback * 10.0
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    finite_mask = np.isfinite(d)

    nearest = np.min(d, axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, fallback)
    nearest = np.maximum(nearest, eps)

    masked_d = np.where(finite_mask, d, np.nan)
    median = np.nanmedian(masked_d, axis=1)
    median = np.nan_to_num(
        median,
        nan=fallback,
        posinf=fallback,
        neginf=fallback,
    )
    median = np.maximum(median, eps)

    local_scale = np.sqrt(nearest * median)
    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    endpoint_scale = np.maximum(endpoint_scale, eps)

    proximity = np.exp(-d / endpoint_scale)
    proximity = np.nan_to_num(
        proximity,
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

    valid_positive = np.isfinite(distance_matrix) & (distance_matrix > 0)
    valid_positive &= ~np.eye(n, dtype=bool)

    inverse_distance = np.zeros((n, n), dtype=float)
    np.divide(
        1.0,
        distance_matrix,
        out=inverse_distance,
        where=valid_positive,
    )

    row_scale = np.max(inverse_distance, axis=1, keepdims=True)
    local_preference = np.divide(
        inverse_distance,
        row_scale,
        out=np.zeros_like(inverse_distance),
        where=row_scale > 0,
    )

    local_preference = 0.5 * (
        local_preference + local_preference.T
    )

    heuristic = (
        0.55 * proximity
        + 0.25 * mutual_rank
        + 0.20 * local_preference
    )

    heuristic = 0.5 * (heuristic + heuristic.T)
    heuristic[~finite_nonnegative] = 0.0
    heuristic = np.nan_to_num(
        heuristic,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    heuristic = np.maximum(heuristic, 0.0)
    np.fill_diagonal(heuristic, 0.0)

    return heuristic
