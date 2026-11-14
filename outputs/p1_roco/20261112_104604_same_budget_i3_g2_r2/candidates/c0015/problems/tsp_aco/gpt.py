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

    positive_mask = valid & (distance_matrix > 0.0)
    positive = distance_matrix[positive_mask]

    eps = np.finfo(float).eps

    if positive.size:
        q10, q50, q90 = np.percentile(positive, [10.0, 50.0, 90.0])
        robust_scale = max(float(q50), eps)
        spread = max(float(q90 - q10), robust_scale * 0.1, eps)
        lower_clip = max(float(q10) * 0.25, robust_scale * 1e-3, eps)
        upper_clip = max(float(q90), lower_clip)
    else:
        robust_scale = 1.0
        spread = 1.0
        lower_clip = eps
        upper_clip = 1.0

    positive_values = np.where(
        positive_mask,
        np.clip(distance_matrix, lower_clip, upper_clip),
        robust_scale,
    )

    row_scale = np.full(n, robust_scale, dtype=float)
    col_scale = np.full(n, robust_scale, dtype=float)
    second_nearest = np.full(n, robust_scale, dtype=float)

    for i in range(n):
        row_positive = distance_matrix[i, valid[i]]
        row_positive = row_positive[row_positive > 0.0]

        if row_positive.size:
            clipped_row = np.clip(row_positive, lower_clip, upper_clip)
            row_scale[i] = max(float(np.median(clipped_row)), lower_clip)

            if row_positive.size >= 2:
                second_nearest[i] = float(
                    np.partition(row_positive, 1)[1]
                )
            else:
                second_nearest[i] = float(row_positive[0])

        col_positive = distance_matrix[valid[:, i], i]
        col_positive = col_positive[col_positive > 0.0]

        if col_positive.size:
            clipped_col = np.clip(col_positive, lower_clip, upper_clip)
            col_scale[i] = max(float(np.median(clipped_col)), lower_clip)

    local_scale = np.sqrt(
        np.maximum(row_scale[:, None], lower_clip)
        * np.maximum(col_scale[None, :], lower_clip)
    )

    safe_distance = np.where(valid, distance_matrix, robust_scale)
    clipped_distance = np.clip(safe_distance, lower_clip, upper_clip)

    zero_edge = valid & (distance_matrix == 0.0)

    inverse = 1.0 / clipped_distance
    inverse = np.where(valid, inverse, 0.0)
    inverse_values = inverse[valid]
    if inverse_values.size:
        inverse_low, inverse_high = np.percentile(
            inverse_values, [5.0, 95.0]
        )
        inverse = np.clip(
            (inverse - inverse_low)
            / (inverse_high - inverse_low + eps),
            0.0,
            1.0,
        )
    inverse[zero_edge] = 1.0

    local_affinity = np.exp(
        -clipped_distance / (local_scale + eps)
    )
    local_affinity = np.where(valid, local_affinity, 0.0)
    local_affinity[zero_edge] = 1.0

    row_preference = np.exp(
        -clipped_distance / (row_scale[:, None] + eps)
    )
    col_preference = np.exp(
        -clipped_distance / (col_scale[None, :] + eps)
    )
    mutual_preference = np.sqrt(
        np.maximum(row_preference, 0.0)
        * np.maximum(col_preference, 0.0)
    )
    mutual_preference = np.where(valid, mutual_preference, 0.0)
    mutual_preference[zero_edge] = 1.0

    second = np.maximum(second_nearest[:, None], lower_clip)
    savings = (second - clipped_distance) / (
        second + clipped_distance + spread + eps
    )
    savings = np.clip(savings, 0.0, 1.0)
    savings = np.where(valid, savings, 0.0)
    savings[zero_edge] = 1.0

    score = (
        0.42 * inverse
        + 0.28 * local_affinity
        + 0.20 * mutual_preference
        + 0.10 * savings
    )

    score = np.where(valid, score, 0.0)
    maximum = float(np.max(score[valid])) if np.any(valid) else 0.0

    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
