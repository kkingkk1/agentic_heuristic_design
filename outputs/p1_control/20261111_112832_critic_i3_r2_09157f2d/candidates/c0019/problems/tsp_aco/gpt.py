import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square 2D array")

    n, m = distance_matrix.shape
    if n != m:
        raise ValueError("distance_matrix must be a square 2D array")

    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid_edges = np.isfinite(distance_matrix)
    off_diagonal = ~np.eye(n, dtype=bool)
    valid_edges &= off_diagonal

    d = np.asarray(distance_matrix, dtype=float).copy()
    d = np.maximum(d, 0.0)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    finite_max = float(np.max(finite_values))
    large_value = max(finite_max, 1.0) * 10.0 + 1.0
    d[~np.isfinite(d)] = large_value
    np.fill_diagonal(d, large_value)

    usable_values = d[valid_edges]
    if usable_values.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(usable_values))
    if not np.isfinite(scale) or scale <= 0.0:
        positive_values = usable_values[usable_values > 0.0]
        scale = float(np.mean(positive_values)) if positive_values.size else 1.0

    epsilon = max(scale * 1e-12, np.finfo(float).eps)

    score = 1.0 / (d + epsilon)
    score[~valid_edges] = 0.0

    nearest_indices = np.argmin(d, axis=1)
    nearest_valid = valid_edges[
        np.arange(n), nearest_indices
    ]

    mutual = np.zeros((n, n), dtype=float)
    rows = np.arange(n)
    for i in rows[nearest_valid]:
        j = int(nearest_indices[i])
        if nearest_valid[j] and nearest_indices[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    score *= 1.0 + 0.15 * mutual
    score = 0.5 * (score + score.T)
    score[~(valid_edges | valid_edges.T)] = 0.0

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    np.fill_diagonal(score, 0.0)
    return score
