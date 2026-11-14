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
    positive = values[values > 0.0]

    if positive.size:
        global_scale = float(np.median(positive))
        if not np.isfinite(global_scale) or global_scale <= 0.0:
            global_scale = float(np.mean(positive))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    row_scale = np.full(n, global_scale, dtype=float)
    col_scale = np.full(n, global_scale, dtype=float)
    row_broad_scale = np.full(n, global_scale, dtype=float)

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)
    local_affinity = np.zeros((n, n), dtype=float)
    column_affinity = np.zeros((n, n), dtype=float)
    broad_affinity = np.zeros((n, n), dtype=float)
    global_affinity = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue

        row_values = distance_matrix[i, indices]
        ordered = np.sort(row_values)

        local_values = ordered[: min(4, ordered.size)]
        local_positive = local_values[local_values > 0.0]
        if local_positive.size:
            scale = float(np.median(local_positive))
        else:
            scale = global_scale
        if not np.isfinite(scale) or scale <= 0.0:
            scale = global_scale
        row_scale[i] = scale

        broad = float(np.percentile(row_values, 75.0))
        if not np.isfinite(broad) or broad <= 0.0:
            broad = global_scale
        row_broad_scale[i] = broad

        order = indices[np.argsort(row_values, kind="mergesort")]
        count = max(1, indices.size - 1)

        for rank, j in enumerate(order):
            row_rank[i, j] = 1.0 - float(rank) / float(count)

        local_affinity[i, indices] = np.exp(
            -row_values / (row_scale[i] + eps)
        )
        broad_affinity[i, indices] = np.exp(
            -row_values / (row_broad_scale[i] + eps)
        )

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size == 0:
            continue

        col_values = distance_matrix[indices, j]
        ordered = np.sort(col_values)

        local_values = ordered[: min(4, ordered.size)]
        local_positive = local_values[local_values > 0.0]
        if local_positive.size:
            scale = float(np.median(local_positive))
        else:
            scale = global_scale
        if not np.isfinite(scale) or scale <= 0.0:
            scale = global_scale
        col_scale[j] = scale

        order = indices[np.argsort(col_values, kind="mergesort")]
        count = max(1, indices.size - 1)

        for rank, i in enumerate(order):
            col_rank[i, j] = 1.0 - float(rank) / float(count)

        column_affinity[indices, j] = np.exp(
            -col_values / (col_scale[j] + eps)
        )

    global_affinity[valid] = np.exp(
        -distance_matrix[valid] / (global_scale + eps)
    )

    multi_scale = (
        0.25 * global_affinity
        + 0.30 * local_affinity
        + 0.18 * column_affinity
        + 0.27 * broad_affinity
    )
    multi_scale[~valid] = 0.0

    row_max = np.maximum(np.max(multi_scale, axis=1, keepdims=True), eps)
    col_max = np.maximum(np.max(multi_scale, axis=0, keepdims=True), eps)

    row_normalized = multi_scale / row_max
    col_normalized = multi_scale / col_max

    endpoint_competition = (
        0.72 * row_normalized
        + 0.28 * col_normalized
    )

    bidirectional_rank = 0.5 * (row_rank + col_rank)
    diversity_gate = np.sqrt(
        np.maximum(row_rank, 0.0) * np.maximum(col_rank, 0.0)
    )

    best_out = np.full(n, np.inf, dtype=float)
    second_out = np.full(n, np.inf, dtype=float)
    best_in = np.full(n, np.inf, dtype=float)
    second_in = np.full(n, np.inf, dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            ordered = np.sort(distance_matrix[i, indices])
            best_out[i] = ordered[0]
            if ordered.size > 1:
                second_out[i] = ordered[1]

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            ordered = np.sort(distance_matrix[indices, j])
            best_in[j] = ordered[0]
            if ordered.size > 1:
                second_in[j] = ordered[1]

    continuation = np.zeros((n, n), dtype=float)
    regret = np.zeros((n, n), dtype=float)

    for i in range(n):
        outgoing = np.flatnonzero(valid[i])

        for j in outgoing:
            candidate = float(distance_matrix[i, j])

            alternative_out = outgoing[outgoing != j]
            if alternative_out.size:
                alternative_out_cost = float(
                    np.min(distance_matrix[i, alternative_out])
                )
                out_regret = max(
                    alternative_out_cost - candidate, 0.0
                ) / (alternative_out_cost + candidate + eps)
            else:
                out_regret = 0.0

            incoming = np.flatnonzero(valid[:, j])
            alternative_in = incoming[incoming != i]
            if alternative_in.size:
                alternative_in_cost = float(
                    np.min(distance_matrix[alternative_in, j])
                )
                in_regret = max(
                    alternative_in_cost - candidate, 0.0
                ) / (alternative_in_cost + candidate + eps)
            else:
                in_regret = 0.0

            regret[i, j] = 0.5 * (out_regret + in_regret)

            future_out = best_out[j] if np.isfinite(best_out[j]) else global_scale
            future_out_2 = (
                second_out[j]
                if np.isfinite(second_out[j])
                else future_out
            )

            future_in = best_in[i] if np.isfinite(best_in[i]) else global_scale
            future_in_2 = (
                second_in[i]
                if np.isfinite(second_in[i])
                else future_in
            )

            delayed_cost = (
                candidate
                + 0.48 * future_out
                + 0.20 * future_out_2
                + 0.22 * future_in
                + 0.10 * future_in_2
            )

            continuation[i, j] = 1.0 / (
                1.0 + delayed_cost / (global_scale + eps)
            )

    continuation_max = np.maximum(
        np.max(continuation, axis=1, keepdims=True),
        eps,
    )
    continuation_normalized = continuation / continuation_max

    score = (
        0.27 * endpoint_competition
        + 0.20 * multi_scale
        + 0.16 * bidirectional_rank
        + 0.13 * diversity_gate
        + 0.19 * continuation_normalized
        + 0.05 * regret
    )

    score[~valid] = 0.0
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    return score
