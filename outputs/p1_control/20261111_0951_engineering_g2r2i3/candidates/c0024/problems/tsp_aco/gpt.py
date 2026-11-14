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
    invalid_value = max(fallback * 10.0, np.finfo(float).eps)

    d[~np.isfinite(d)] = invalid_value
    d[d <= 0] = invalid_value
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps

    inverse_distance = 1.0 / np.maximum(d, eps)
    row_inverse = inverse_distance / np.maximum(
        np.max(inverse_distance, axis=1, keepdims=True),
        eps,
    )
    col_inverse = inverse_distance / np.maximum(
        np.max(inverse_distance, axis=0, keepdims=True),
        eps,
    )
    distance_score = np.sqrt(row_inverse * col_inverse)

    ranks = np.argsort(np.argsort(d, axis=1), axis=1).astype(float)
    row_rank_score = 1.0 / (1.0 + ranks)
    row_rank_score /= np.maximum(
        np.max(row_rank_score, axis=1, keepdims=True),
        eps,
    )

    reverse_ranks = np.argsort(np.argsort(d, axis=0), axis=0).astype(float)
    col_rank_score = 1.0 / (1.0 + reverse_ranks)
    col_rank_score /= np.maximum(
        np.max(col_rank_score, axis=0, keepdims=True),
        eps,
    )

    rank_score = np.sqrt(row_rank_score * col_rank_score)

    heuristic = 0.70 * distance_score + 0.30 * rank_score
    heuristic = np.nan_to_num(
        heuristic,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    heuristic = np.maximum(heuristic, 0.0)
    np.fill_diagonal(heuristic, 0.0)

    return heuristic
