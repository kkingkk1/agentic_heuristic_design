import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n == 0:
        return result

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(distance_matrix) & (distance_matrix >= 0.0)
    valid &= off_diagonal

    if not np.any(valid):
        return result

    valid_values = distance_matrix[valid]
    positive_values = valid_values[valid_values > 0.0]

    eps = np.finfo(float).eps
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0

    max_valid = float(np.max(valid_values))
    sentinel = max(max_valid, global_scale, 1.0) * 16.0

    distances = np.where(valid, distance_matrix, sentinel)

    row_scales = np.empty((n, 1), dtype=float)
    row_medians = np.empty((n, 1), dtype=float)
    row_mads = np.empty((n, 1), dtype=float)

    for i in range(n):
        values = distance_matrix[valid[i]]
        if values.size == 0:
            row_scales[i, 0] = global_scale
            row_medians[i, 0] = global_scale
            row_mads[i, 0] = global_scale
            continue

        positive = values[values > 0.0]
        if positive.size:
            ordered = np.sort(positive)
            k = min(3, ordered.size)
            local_scale = float(np.median(ordered[:k]))
        else:
            local_scale = global_scale

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))

        row_scales[i, 0] = max(local_scale, eps)
        row_medians[i, 0] = median
        row_mads[i, 0] = max(1.4826 * mad, local_scale * 0.25, eps)

    row_affinity = np.exp(-distances / (row_scales + eps))
    row_affinity[~valid] = 0.0

    column_scales = row_scales.T
    column_affinity = np.exp(-distances / (column_scales + eps))
    column_affinity[~valid] = 0.0

    symmetric_affinity = np.sqrt(row_affinity * column_affinity)

    contrast = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                (distances - row_medians) / (row_mads + eps),
                -40.0,
                40.0,
            )
        )
    )
    contrast[~valid] = 0.0

    savings = np.zeros((n, n), dtype=float)
    if n > 2:
        for i in range(n):
            values = distance_matrix[i, valid[i]]
            if values.size < 2:
                continue

            nearest_two = np.partition(values, 1)[:2]
            second_nearest = float(np.max(nearest_two))

            row_savings = (
                (second_nearest - distances[i])
                / (second_nearest + distances[i] + eps)
            )
            savings[i] = np.maximum(row_savings, 0.0)

        savings[~valid] = 0.0

    score = (
        0.50 * row_affinity
        + 0.25 * symmetric_affinity
        + 0.20 * contrast
        + 0.05 * savings
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > eps,
    )

    return score
