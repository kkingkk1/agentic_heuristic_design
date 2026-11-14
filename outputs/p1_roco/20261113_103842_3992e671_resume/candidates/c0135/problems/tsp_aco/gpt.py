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

    finite_values = distances[valid]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    if np.all(finite_values == 0.0):
        result = np.ones((n, n), dtype=float)
        np.fill_diagonal(result, 0.0)
        return result

    positive_values = finite_values[finite_values > 0.0]
    scale = float(np.median(positive_values)) if positive_values.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe_distances = np.where(valid, distances, np.inf)

    row_best = np.min(safe_distances, axis=1)
    col_best = np.min(safe_distances, axis=0)

    row_fallback = (
        float(np.median(row_best[np.isfinite(row_best)]))
        if np.any(np.isfinite(row_best))
        else scale
    )
    col_fallback = (
        float(np.median(col_best[np.isfinite(col_best)]))
        if np.any(np.isfinite(col_best))
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
    distance_score = 1.0 / (1.0 + normalized * normalized)

    eps = np.finfo(float).eps
    row_local = np.divide(
        row_best[:, None],
        row_best[:, None] + distances + 0.5 * scale + eps,
    )
    col_local = np.divide(
        col_best[None, :],
        col_best[None, :] + distances + 0.5 * scale + eps,
    )
    local_score = 0.5 * (row_local + col_local)
    local_score = np.clip(local_score, 0.0, 1.0)

    scores = distance_score * (0.78 + 0.22 * local_score)

    nearest = np.argmin(safe_distances, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = int(nearest[i])
        if j != i and valid[i, j] and valid[j, i] and nearest[j] == i:
            mutual[i, j] = 1.0

    scores *= 1.0 + 0.12 * mutual

    out_degree = np.sum(valid, axis=1)
    in_degree = np.sum(valid, axis=0)
    max_degree = max(n - 1, 1)

    out_bonus = np.clip(
        np.sqrt(max_degree / np.maximum(out_degree, 1)),
        1.0,
        1.25,
    )
    in_bonus = np.clip(
        np.sqrt(max_degree / np.maximum(in_degree, 1)),
        1.0,
        1.25,
    )

    sparsity_factor = (
        1.0
        + 0.04 * (out_bonus[:, None] - 1.0)
        + 0.04 * (in_bonus[None, :] - 1.0)
    )
    scores *= sparsity_factor

    scores[~valid] = 0.0
    scores[~np.isfinite(scores)] = 0.0
    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
