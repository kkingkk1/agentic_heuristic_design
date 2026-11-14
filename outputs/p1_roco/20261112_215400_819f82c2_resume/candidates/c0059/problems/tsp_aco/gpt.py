import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")

    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.maximum(distance_matrix, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    positive_values = finite_values[finite_values > 0.0]

    eps = np.finfo(float).eps
    if positive_values.size == 0:
        result = np.ones((n, n), dtype=float)
        np.fill_diagonal(result, 0.0)
        return result

    global_scale = float(np.median(positive_values))
    global_scale = max(global_scale, eps)

    k = min(n - 1, max(2, int(np.ceil(np.sqrt(n)))))

    sorted_rows = np.sort(d, axis=1)
    local_scale = sorted_rows[:, k - 1]
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    normalized_distance = d / global_scale
    global_affinity = np.exp(-np.minimum(normalized_distance, 60.0))

    row_affinity = np.exp(
        -np.minimum(d / local_scale[:, None], 60.0)
    )
    col_affinity = row_affinity.T

    row_order = np.argsort(d, axis=1)
    row_rank = np.empty_like(row_order)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)[None, :]

    col_order = np.argsort(d, axis=0)
    col_rank = np.empty_like(col_order)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]

    row_rank_score = 1.0 / (1.0 + row_rank.astype(float))
    col_rank_score = 1.0 / (1.0 + col_rank.astype(float))

    reciprocal_rank = np.sqrt(row_rank_score * col_rank_score)

    row_support = (row_rank < k).astype(float)
    col_support = (col_rank < k).astype(float)
    reciprocal_support = np.sqrt(row_support * col_support)

    row_competition = np.sum(
        np.exp(-np.minimum(d / local_scale[:, None], 60.0)),
        axis=1,
    ) - 1.0
    col_competition = np.sum(
        np.exp(-np.minimum(d / local_scale[None, :], 60.0)),
        axis=0,
    ) - 1.0

    row_pressure = 1.0 / np.sqrt(1.0 + np.maximum(row_competition, 0.0))
    col_pressure = 1.0 / np.sqrt(1.0 + np.maximum(col_competition, 0.0))
    scarcity = np.sqrt(row_pressure[:, None] * col_pressure[None, :])

    local_quality = np.sqrt(row_affinity * col_affinity)
    mutual_quality = 0.5 * (row_affinity + col_affinity)

    score = (
        0.30 * global_affinity
        + 0.28 * local_quality
        + 0.18 * mutual_quality
        + 0.14 * reciprocal_rank
        + 0.06 * reciprocal_support
        + 0.04 * scarcity
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
