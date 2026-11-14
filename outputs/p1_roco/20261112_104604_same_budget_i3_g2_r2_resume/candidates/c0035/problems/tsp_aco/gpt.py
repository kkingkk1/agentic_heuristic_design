import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = (
        np.isfinite(distance_matrix)
        & (distance_matrix >= 0.0)
        & off_diagonal
    )

    if not np.any(valid):
        return score

    eps = np.finfo(float).eps
    values = distance_matrix[valid]
    positive_values = values[values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
        if not np.isfinite(global_scale) or global_scale <= 0.0:
            global_scale = float(np.mean(positive_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    row_scale = np.full(n, global_scale, dtype=float)
    col_scale = np.full(n, global_scale, dtype=float)

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)
    row_affinity = np.zeros((n, n), dtype=float)
    col_affinity = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue

        row_values = distance_matrix[i, indices]
        ordered = np.sort(row_values)
        local_count = min(4, ordered.size)
        local_values = ordered[:local_count]
        positive_local = local_values[local_values > 0.0]

        if positive_local.size:
            scale = float(np.median(positive_local))
        else:
            scale = global_scale

        if not np.isfinite(scale) or scale <= 0.0:
            scale = global_scale

        row_scale[i] = scale
        order = indices[np.argsort(row_values, kind="mergesort")]

        for rank, j in enumerate(order):
            row_rank[i, j] = 1.0 / (1.0 + float(rank))

        row_affinity[i, indices] = np.exp(
            -row_values / (row_scale[i] + eps)
        )

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size == 0:
            continue

        col_values = distance_matrix[indices, j]
        ordered = np.sort(col_values)
        local_count = min(4, ordered.size)
        local_values = ordered[:local_count]
        positive_local = local_values[local_values > 0.0]

        if positive_local.size:
            scale = float(np.median(positive_local))
        else:
            scale = global_scale

        if not np.isfinite(scale) or scale <= 0.0:
            scale = global_scale

        col_scale[j] = scale
        order = indices[np.argsort(col_values, kind="mergesort")]

        for rank, i in enumerate(order):
            col_rank[i, j] = 1.0 / (1.0 + float(rank))

        col_affinity[indices, j] = np.exp(
            -col_values / (col_scale[j] + eps)
        )

    global_affinity = np.zeros((n, n), dtype=float)
    global_affinity[valid] = np.exp(
        -distance_matrix[valid] / (global_scale + eps)
    )

    broad_scale = np.zeros(n, dtype=float)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            broad_scale[i] = float(np.percentile(
                distance_matrix[i, indices], 70
            ))
            if not np.isfinite(broad_scale[i]) or broad_scale[i] <= 0.0:
                broad_scale[i] = global_scale
        else:
            broad_scale[i] = global_scale

    broad_affinity = np.zeros((n, n), dtype=float)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            broad_affinity[i, indices] = np.exp(
                -distance_matrix[i, indices] / (broad_scale[i] + eps)
            )

    multi_scale = (
        0.30 * global_affinity
        + 0.35 * row_affinity
        + 0.20 * col_affinity
        + 0.15 * broad_affinity
    )

    rank_competition = 0.5 * (row_rank + col_rank)

    best_out = np.full(n, np.inf, dtype=float)
    best_in = np.full(n, np.inf, dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            best_out[i] = float(np.min(distance_matrix[i, indices]))

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            best_in[j] = float(np.min(distance_matrix[indices, j]))

    continuation = np.zeros((n, n), dtype=float)
    regret = np.zeros((n, n), dtype=float)

    for i in range(n):
        outgoing = np.flatnonzero(valid[i])

        for j in outgoing:
            alternatives_out = outgoing[outgoing != j]
            incoming = np.flatnonzero(valid[:, j])
            alternatives_in = incoming[incoming != i]

            candidate = float(distance_matrix[i, j])

            if alternatives_out.size:
                alternative_out = float(
                    np.min(distance_matrix[i, alternatives_out])
                )
                out_regret = max(
                    alternative_out - candidate, 0.0
                ) / (alternative_out + candidate + eps)
            else:
                out_regret = 0.0

            if alternatives_in.size:
                alternative_in = float(
                    np.min(distance_matrix[alternatives_in, j])
                )
                in_regret = max(
                    alternative_in - candidate, 0.0
                ) / (alternative_in + candidate + eps)
            else:
                in_regret = 0.0

            regret[i, j] = 0.5 * (out_regret + in_regret)

            future_out = best_out[j]
            if not np.isfinite(future_out):
                future_out = global_scale

            future_in = best_in[i]
            if not np.isfinite(future_in):
                future_in = global_scale

            continuation[i, j] = 1.0 / (
                1.0
                + (
                    candidate
                    + 0.5 * future_out
                    + 0.5 * future_in
                ) / (global_scale + eps)
            )

    row_max = np.max(multi_scale, axis=1, keepdims=True)
    row_max = np.maximum(row_max, eps)
    row_normalized = multi_scale / row_max

    col_max = np.max(multi_scale, axis=0, keepdims=True)
    col_max = np.maximum(col_max, eps)
    col_normalized = multi_scale / col_max

    endpoint_normalized = 0.55 * row_normalized + 0.45 * col_normalized

    continuation_max = np.max(continuation, axis=1, keepdims=True)
    continuation_max = np.maximum(continuation_max, eps)
    continuation_normalized = continuation / continuation_max

    score = (
        0.35 * endpoint_normalized
        + 0.20 * multi_scale
        + 0.18 * rank_competition
        + 0.17 * continuation_normalized
        + 0.10 * regret
    )

    score[~valid] = 0.0
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    return score
