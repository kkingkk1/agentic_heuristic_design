import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    d = distance_matrix.copy()
    valid = np.isfinite(d)
    np.fill_diagonal(valid, False)

    finite_values = d[valid]
    if finite_values.size == 0:
        return np.zeros_like(d, dtype=float)

    max_distance = np.max(finite_values)
    d[~valid] = max_distance

    positive = d[(valid) & (d > 0)]
    epsilon = max(np.finfo(float).eps, np.median(positive) * 1e-12) if positive.size else np.finfo(float).eps

    closeness = 1.0 / (d + epsilon)
    closeness[~valid] = 0.0

    row_scale = np.max(closeness, axis=1, keepdims=True)
    row_scale[row_scale == 0.0] = 1.0
    row_attractiveness = closeness / row_scale

    col_scale = np.max(closeness, axis=0, keepdims=True)
    col_scale[col_scale == 0.0] = 1.0
    col_attractiveness = closeness / col_scale

    nearest = np.argmax(np.where(valid, d, np.inf), axis=1)
    mutual_bonus = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if 0 <= j < n and nearest[j] == i and i != j:
            mutual_bonus[i, j] = 1.0

    score = (
        0.60 * row_attractiveness
        + 0.25 * col_attractiveness
        + 0.15 * mutual_bonus
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    row_max[row_max == 0.0] = 1.0
    return score / row_max
