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

    finite_nonnegative = np.isfinite(distance_matrix) & (distance_matrix >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = (
        distance_matrix[valid] + distance_matrix.T[valid]
    ) * 0.5
    np.fill_diagonal(distances, np.inf)

    valid_values = distances[np.isfinite(distances)]
    if valid_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = valid_values[valid_values > 0.0]
    if positive_values.size:
        fallback_scale = float(np.median(positive_values))
    else:
        fallback_scale = 1.0

    nearest = np.min(distances, axis=1)
    finite_nearest = nearest[np.isfinite(nearest)]
    if finite_nearest.size:
        positive_nearest = finite_nearest[finite_nearest > 0.0]
        scale = (
            float(np.median(positive_nearest))
            if positive_nearest.size
            else fallback_scale
        )
    else:
        scale = fallback_scale

    scale = max(scale, np.finfo(float).eps)

    k = min(3, n - 1)
    local_scale = np.full(n, scale, dtype=float)

    for i in range(n):
        row = distances[i, np.isfinite(distances[i])]
        if row.size:
            nearest_values = np.partition(row, min(k, row.size) - 1)
            local_scale[i] = float(
                np.mean(nearest_values[: min(k, row.size)])
            )

    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    row_rank = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_valid = valid[i]
        if np.any(row_valid):
            row_distances = distances[i]
            row_rank[i, row_valid] = (
                np.sum(
                    row_distances[row_valid, None]
                    < row_distances[None, row_valid],
                    axis=0,
                )
                + 1.0
            )

    symmetric_rank = 0.5 * (row_rank + row_rank.T)
    rank_score = np.zeros((n, n), dtype=float)
    rank_score[valid] = 1.0 / (1.0 + symmetric_rank[valid])

    nearest_neighbor = np.argmin(distances, axis=1)
    mutual = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest_neighbor[i]
        if np.isfinite(distances[i, j]) and nearest_neighbor[j] == i:
            mutual[i, j] = 1.0

    pair_local_scale = np.sqrt(
        np.outer(local_scale, local_scale)
    )
    robust_scale = 0.75 * scale + 0.25 * pair_local_scale

    normalized_distance = distances / np.maximum(
        robust_scale, np.finfo(float).eps
    )

    short_edge_score = np.zeros((n, n), dtype=float)
    short_edge_score[valid] = 1.0 / (
        1.0 + normalized_distance[valid] ** 2
    )

    directional_score = np.zeros((n, n), dtype=float)
    directional_score[valid] = 0.5 * (
        np.exp(-distances[valid] / local_scale[np.newaxis, :].repeat(n, axis=0)[valid])
        + np.exp(-distances[valid] / local_scale[:, np.newaxis].repeat(n, axis=1)[valid])
    )

    scores = (
        0.55 * short_edge_score
        + 0.20 * rank_score
        + 0.15 * directional_score
        + 0.10 * mutual
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores, initial=0.0))
    if maximum > 0.0:
        scores /= maximum

    return scores
