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

    d = np.nan_to_num(
        distance_matrix,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    ).copy()

    finite_positive = np.isfinite(d) & (d > 0)
    if not np.any(finite_positive):
        return np.zeros_like(d, dtype=float)

    positive_values = d[finite_positive]
    fallback = float(np.max(positive_values))
    invalid_value = fallback * 10.0

    d[~np.isfinite(d)] = invalid_value
    d[d <= 0] = invalid_value
    np.fill_diagonal(d, np.inf)

    inverse_distance = 1.0 / np.maximum(d, np.finfo(float).eps)
    inverse_max = np.max(inverse_distance, axis=1, keepdims=True)
    inverse_distance = np.divide(
        inverse_distance,
        inverse_max,
        out=np.zeros_like(inverse_distance),
        where=inverse_max > 0,
    )

    ranks = np.argsort(np.argsort(d, axis=1), axis=1)
    rank_score = 1.0 / (1.0 + ranks.astype(float))
    rank_max = np.max(rank_score, axis=1, keepdims=True)
    rank_score = np.divide(
        rank_score,
        rank_max,
        out=np.zeros_like(rank_score),
        where=rank_max > 0,
    )

    nearest = np.argmin(d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if nearest[j] == i:
            mutual[i, j] = 1.0

    row_second = np.partition(d, 1, axis=1)[:, 1]
    savings = row_second[:, None] + row_second[None, :] - d
    savings = np.maximum(savings, 0.0)

    savings_max = np.max(savings, axis=1, keepdims=True)
    savings_score = np.divide(
        savings,
        savings_max,
        out=np.zeros_like(savings),
        where=savings_max > 0,
    )

    heuristic = (
        0.55 * inverse_distance
        + 0.20 * rank_score
        + 0.15 * savings_score
        + 0.10 * mutual
    )

    heuristic = np.nan_to_num(
        heuristic,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    heuristic = np.maximum(heuristic, 0.0)
    np.fill_diagonal(heuristic, 0.0)

    return heuristic
