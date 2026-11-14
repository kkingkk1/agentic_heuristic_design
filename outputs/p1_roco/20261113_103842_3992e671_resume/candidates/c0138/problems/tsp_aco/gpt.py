import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(distance_matrix)
    valid = finite & off_diagonal

    distances = np.zeros_like(distance_matrix, dtype=float)
    distances[finite] = np.maximum(distance_matrix[finite], 0.0)

    valid_values = distances[valid]
    if valid_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = valid_values[valid_values > 0.0]
    if positive_values.size == 0:
        result = np.ones((n, n), dtype=float)
        np.fill_diagonal(result, 0.0)
        return result

    scale = float(np.median(positive_values))
    scale = max(scale, np.finfo(float).eps)

    safe_distances = np.where(valid, distances, np.inf)

    row_best = np.min(safe_distances, axis=1)
    col_best = np.min(safe_distances, axis=0)

    finite_row_best = row_best[np.isfinite(row_best)]
    finite_col_best = col_best[np.isfinite(col_best)]

    row_fallback = (
        float(np.median(finite_row_best))
        if finite_row_best.size
        else scale
    )
    col_fallback = (
        float(np.median(finite_col_best))
        if finite_col_best.size
        else scale
    )

    row_best = np.where(np.isfinite(row_best), row_best, row_fallback)
    col_best = np.where(np.isfinite(col_best), col_best, col_fallback)

    normalized = np.divide(
        distances,
        scale,
        out=np.zeros_like(distances),
        where=valid,
    )

    distance_score = 1.0 / (1.0 + np.power(normalized, 1.5))

    eps = np.finfo(float).eps
    row_gap = np.maximum(distances - row_best[:, None], 0.0)
    col_gap = np.maximum(distances - col_best[None, :], 0.0)

    row_local = 1.0 / (
        1.0 + row_gap / (scale + row_best[:, None] + eps)
    )
    col_local = 1.0 / (
        1.0 + col_gap / (scale + col_best[None, :] + eps)
    )

    local_score = 0.5 * (row_local + col_local)
    local_score = np.clip(local_score, 0.0, 1.0)

    scores = distance_score * (0.88 + 0.12 * local_score)

    nearest = np.argmin(safe_distances, axis=1)
    mutual = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = int(nearest[i])
        if (
            j != i
            and valid[i, j]
            and valid[j, i]
            and nearest[j] == i
        ):
            mutual[i, j] = 1.0

    scores *= 1.0 + 0.08 * mutual

    scores[~valid] = 0.0
    scores[~np.isfinite(scores)] = 0.0
    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
