import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    result = np.zeros_like(distance_matrix, dtype=float)

    if n == 0:
        return result

    finite = np.isfinite(distance_matrix)
    valid = finite & (distance_matrix >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    positive = distance_matrix[valid & (distance_matrix > 0.0)]

    if positive.size:
        global_median = float(np.median(positive))
        q25, q75 = np.percentile(positive, [25.0, 75.0])
        global_spread = max(float(q75 - q25), global_median * 0.1)
        global_scale = max(global_median, np.finfo(float).eps)
    else:
        global_median = 1.0
        global_spread = 1.0
        global_scale = 1.0

    row_scale = np.full(n, global_scale, dtype=float)
    col_scale = np.full(n, global_scale, dtype=float)
    second_nearest = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row_values = distance_matrix[i, valid[i]]
        if row_values.size:
            positive_row = row_values[row_values > 0.0]
            if positive_row.size:
                row_scale[i] = max(float(np.median(positive_row)), global_scale * 0.1)

                if positive_row.size >= 2:
                    second_nearest[i] = float(np.partition(positive_row, 1)[1])
                else:
                    second_nearest[i] = float(positive_row[0])
            else:
                row_scale[i] = global_scale
                second_nearest[i] = global_scale

        col_values = distance_matrix[valid[:, i], i]
        positive_col = col_values[col_values > 0.0]
        if positive_col.size:
            col_scale[i] = max(float(np.median(positive_col)), global_scale * 0.1)

    eps = np.finfo(float).eps
    local_scale = np.sqrt(
        np.maximum(row_scale[:, None], eps)
        * np.maximum(col_scale[None, :], eps)
    )

    safe_distance = np.where(valid, distance_matrix, 0.0)

    inverse = 1.0 / (safe_distance + global_scale * 1e-6 + eps)
    inverse /= np.max(inverse[valid]) + eps

    local_affinity = np.exp(-safe_distance / (local_scale + eps))
    local_affinity *= valid

    row_preference = np.exp(
        -safe_distance / (row_scale[:, None] + eps)
    )
    col_preference = np.exp(
        -safe_distance / (col_scale[None, :] + eps)
    )
    mutual_preference = np.sqrt(row_preference * col_preference)
    mutual_preference *= valid

    savings = (
        (second_nearest[:, None] - safe_distance)
        / (second_nearest[:, None] + safe_distance + global_spread + eps)
    )
    savings = np.maximum(savings, 0.0) * valid

    score = (
        0.40 * inverse
        + 0.30 * local_affinity
        + 0.20 * mutual_preference
        + 0.10 * savings
    )

    score *= valid
    maximum = np.max(score) if np.any(valid) else 0.0
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
