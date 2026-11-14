import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n == 0:
        return score

    finite = np.isfinite(distance_matrix)
    valid = finite & (distance_matrix >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    eps = np.finfo(float).eps
    positive = distance_matrix[valid & (distance_matrix > 0.0)]

    if positive.size:
        q05, q10, q50, q90, q95 = np.percentile(
            positive, [5.0, 10.0, 50.0, 90.0, 95.0]
        )
        positive_floor = max(float(q05), eps)
        lower_clip = max(float(q10) * 0.25, positive_floor, eps)
        upper_clip = max(float(q95), lower_clip)
        global_scale = max(float(q50), positive_floor)
        spread = max(float(q90 - q10), global_scale * 0.1, eps)
    else:
        positive_floor = eps
        lower_clip = eps
        upper_clip = 1.0
        global_scale = 1.0
        spread = 1.0

    safe_distance = np.where(
        valid,
        np.maximum(distance_matrix, positive_floor),
        global_scale,
    )
    clipped_distance = np.clip(safe_distance, lower_clip, upper_clip)
    zero_edge = valid & (distance_matrix <= eps)

    row_scale = np.full(n, global_scale, dtype=float)
    col_scale = np.full(n, global_scale, dtype=float)
    row_best = np.full(n, global_scale, dtype=float)
    row_second = np.full(n, global_scale, dtype=float)
    col_best = np.full(n, global_scale, dtype=float)
    col_second = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row_values = distance_matrix[i, valid[i]]
        row_positive = row_values[row_values > 0.0]

        if row_positive.size:
            clipped_row = np.clip(row_positive, lower_clip, upper_clip)
            row_scale[i] = max(float(np.median(clipped_row)), lower_clip)
            ordered = np.sort(row_positive)
            row_best[i] = max(float(ordered[0]), positive_floor)
            row_second[i] = (
                max(float(ordered[1]), positive_floor)
                if ordered.size > 1
                else row_best[i]
            )

        col_values = distance_matrix[valid[:, i], i]
        col_positive = col_values[col_values > 0.0]

        if col_positive.size:
            clipped_col = np.clip(col_positive, lower_clip, upper_clip)
            col_scale[i] = max(float(np.median(clipped_col)), lower_clip)
            ordered = np.sort(col_positive)
            col_best[i] = max(float(ordered[0]), positive_floor)
            col_second[i] = (
                max(float(ordered[1]), positive_floor)
                if ordered.size > 1
                else col_best[i]
            )

    row_rank = np.full((n, n), float(n), dtype=float)
    col_rank = np.full((n, n), float(n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[np.argsort(distance_matrix[i, indices], kind="mergesort")]
            row_rank[i, order] = np.arange(1.0, order.size + 1.0)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = indices[
                np.argsort(distance_matrix[indices, j], kind="mergesort")
            ]
            col_rank[order, j] = np.arange(1.0, order.size + 1.0)

    row_count = np.maximum(np.sum(valid, axis=1, keepdims=True), 1.0)
    col_count = np.maximum(np.sum(valid, axis=0, keepdims=True), 1.0)

    outgoing_preference = np.where(
        valid,
        1.0 - (row_rank - 1.0) / row_count,
        0.0,
    )
    incoming_preference = np.where(
        valid,
        1.0 - (col_rank - 1.0) / col_count,
        0.0,
    )
    mutual_preference = np.sqrt(
        np.maximum(outgoing_preference, 0.0)
        * np.maximum(incoming_preference, 0.0)
    )

    local_scale = np.sqrt(
        np.maximum(row_scale[:, None], lower_clip)
        * np.maximum(col_scale[None, :], lower_clip)
    )

    inverse = 1.0 / clipped_distance
    inverse = np.where(valid, inverse, 0.0)
    inverse_values = inverse[valid]

    if inverse_values.size:
        inverse_low, inverse_high = np.percentile(
            inverse_values, [5.0, 95.0]
        )
        inverse = np.clip(
            (inverse - inverse_low) / (inverse_high - inverse_low + eps),
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
    mutual_exponential = np.sqrt(
        np.maximum(row_preference, 0.0)
        * np.maximum(col_preference, 0.0)
    )
    mutual_exponential = np.where(valid, mutual_exponential, 0.0)
    mutual_exponential[zero_edge] = 1.0

    row_regret = np.divide(
        np.maximum(row_second - row_best, 0.0),
        row_second + eps,
        out=np.zeros_like(row_best),
        where=row_second > 0.0,
    )
    col_regret = np.divide(
        np.maximum(col_second - col_best, 0.0),
        col_second + eps,
        out=np.zeros_like(col_best),
        where=col_second > 0.0,
    )
    opportunity = np.sqrt(
        (0.55 + 0.45 * row_regret)[:, None]
        * (0.55 + 0.45 * col_regret)[None, :]
    )
    opportunity = np.where(valid, opportunity, 0.0)

    if n > 2:
        two_step = np.full((n, n), np.nan, dtype=float)

        for i in range(n):
            for j in range(n):
                if not valid[i, j]:
                    continue

                intermediates = valid[i] & valid[:, j]
                intermediates[i] = False
                intermediates[j] = False

                if np.any(intermediates):
                    two_step[i, j] = np.min(
                        safe_distance[i, intermediates]
                        + safe_distance[intermediates, j]
                    )

        direct = np.where(valid, safe_distance, np.nan)
        savings = np.divide(
            two_step - direct,
            two_step + direct + spread + eps,
            out=np.zeros((n, n), dtype=float),
            where=np.isfinite(two_step) & np.isfinite(direct),
        )
        savings = np.clip(savings, 0.0, 1.0)
    else:
        savings = np.zeros((n, n), dtype=float)

    savings[zero_edge] = 1.0

    rank_signal = np.sqrt(
        np.maximum(outgoing_preference, 0.0)
        * np.maximum(incoming_preference, 0.0)
    )

    score = (
        0.36 * inverse
        + 0.24 * local_affinity
        + 0.14 * mutual_exponential
        + 0.10 * rank_signal
        + 0.09 * opportunity
        + 0.07 * savings
    )

    score = np.where(valid, np.maximum(score, 0.0), 0.0)
    maximum = float(np.max(score[valid])) if np.any(valid) else 0.0

    if maximum > 0.0:
        score /= maximum

    score[zero_edge] = 1.0
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
