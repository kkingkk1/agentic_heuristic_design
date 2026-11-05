import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    valid = np.isfinite(distance_matrix) & (distance_matrix >= 0.0)
    np.fill_diagonal(valid, False)

    positive = np.where(valid & (distance_matrix > 0.0), distance_matrix, np.nan)
    fallback = np.nanmax(np.where(np.isfinite(positive), positive, 0.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        return np.zeros_like(distance_matrix, dtype=float)

    distances = np.where(valid, distance_matrix, fallback)
    np.fill_diagonal(distances, fallback)

    eps = np.finfo(float).eps
    inverse = 1.0 / np.maximum(distances, eps)
    np.fill_diagonal(inverse, 0.0)

    row_mean = np.sum(inverse, axis=1) / np.maximum(np.sum(valid, axis=1), 1)
    col_mean = np.sum(inverse, axis=0) / np.maximum(np.sum(valid, axis=0), 1)

    local_distinctiveness = inverse / (
        np.sqrt(np.outer(row_mean, col_mean)) + eps
    )
    local_distinctiveness *= valid

    continuation = np.empty((n, n), dtype=float)
    for i in range(n):
        through = inverse[i, :, None] * inverse[None, :, :]
        through[:, i] = 0.0
        through[:, np.arange(n)] *= valid[i, :, None]
        continuation[i] = np.max(through, axis=0)
    continuation *= valid

    def normalize_rows(values: np.ndarray) -> np.ndarray:
        row_max = np.max(values, axis=1, keepdims=True)
        row_min = np.min(np.where(valid, values, np.inf), axis=1, keepdims=True)
        scale = row_max - row_min
        return np.where(
            valid,
            np.where(scale > eps, (values - row_min) / (scale + eps), 0.0),
            0.0,
        )

    score = (
        0.50 * normalize_rows(inverse)
        + 0.30 * normalize_rows(local_distinctiveness)
        + 0.20 * normalize_rows(continuation)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    return score
