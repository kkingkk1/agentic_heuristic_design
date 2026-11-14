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

    finite_nonnegative = np.isfinite(distance_matrix) & (distance_matrix >= 0.0)
    valid = finite_nonnegative.copy()
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    eps = np.finfo(float).eps
    safe_distance = np.where(valid, distance_matrix, np.inf)

    positive_values = distance_matrix[valid & (distance_matrix > 0.0)]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
        if not np.isfinite(global_scale) or global_scale <= 0.0:
            global_scale = float(np.mean(positive_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    def robust_local_scales(values: np.ndarray) -> np.ndarray:
        scales = np.full(n, global_scale, dtype=float)

        for i in range(n):
            row = values[i]
            finite_row = row[np.isfinite(row) & (row > 0.0)]

            if finite_row.size:
                nearest = np.partition(
                    finite_row, min(2, finite_row.size - 1)
                )[: min(3, finite_row.size)]
                scale = float(np.median(nearest))

                if np.isfinite(scale) and scale > 0.0:
                    scales[i] = scale

        return scales

    row_scale = robust_local_scales(safe_distance)
    col_scale = robust_local_scales(safe_distance.T)

    directed_global = np.zeros((n, n), dtype=float)
    directed_global[valid] = np.exp(
        -distance_matrix[valid] / (global_scale + eps)
    )

    outgoing_affinity = np.zeros((n, n), dtype=float)
    incoming_affinity = np.zeros((n, n), dtype=float)

    outgoing_affinity[valid] = np.exp(
        -distance_matrix[valid]
        / (row_scale[:, None] + eps)[valid]
    )
    incoming_affinity[valid] = np.exp(
        -distance_matrix[valid]
        / (col_scale[None, :] + eps)[valid]
    )

    outgoing_rank = np.zeros((n, n), dtype=float)
    incoming_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[np.argsort(distance_matrix[i, indices])]
            outgoing_rank[i, order] = 1.0 / (
                1.0 + np.arange(order.size, dtype=float)
            )

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = indices[np.argsort(distance_matrix[indices, j])]
            incoming_rank[order, j] = 1.0 / (
                1.0 + np.arange(order.size, dtype=float)
            )

    local_affinity = (
        0.5 * outgoing_affinity
        + 0.5 * incoming_affinity
    )
    local_rank = 0.5 * (outgoing_rank + incoming_rank)

    local_evidence = 0.7 * local_affinity + 0.3 * local_rank

    row_order = np.argsort(safe_distance, axis=1)
    col_order = np.argsort(safe_distance, axis=0)

    row_best_index = row_order[:, 0]
    row_best_value = safe_distance[
        np.arange(n), row_best_index
    ]

    if n > 1:
        row_second_value = safe_distance[
            np.arange(n), row_order[:, 1]
        ]
    else:
        row_second_value = np.full(n, np.inf)

    col_best_index = col_order[0, :]
    col_best_value = safe_distance[
        col_best_index, np.arange(n)
    ]

    if n > 1:
        col_second_value = safe_distance[
            col_order[1, :], np.arange(n)
        ]
    else:
        col_second_value = np.full(n, np.inf)

    alternative_out = np.where(
        row_best_index[None, :] == np.arange(n)[:, None],
        row_second_value[:, None],
        row_best_value[:, None],
    )

    alternative_in = np.where(
        col_best_index[None, :] == np.arange(n)[:, None],
        col_second_value[None, :],
        col_best_value[None, :],
    )

    candidate = distance_matrix
    alternative_sum = alternative_out + alternative_in

    finite_alternatives = np.isfinite(alternative_sum)
    detour = np.zeros((n, n), dtype=float)

    detour[valid & finite_alternatives] = np.clip(
        (
            alternative_sum[valid & finite_alternatives]
            - 2.0 * candidate[valid & finite_alternatives]
        )
        / (alternative_sum[valid & finite_alternatives] + eps),
        0.0,
        1.0,
    )

    bottleneck = np.zeros((n, n), dtype=float)
    bottleneck_argument = (
        alternative_sum / (2.0 * global_scale + eps) - 1.0
    )
    bottleneck[valid & finite_alternatives] = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                bottleneck_argument[valid & finite_alternatives],
                -30.0,
                30.0,
            )
        )
    )

    continuation = np.zeros((n, n), dtype=float)
    for i in range(n):
        destinations = np.flatnonzero(valid[i])

        for j in destinations:
            next_edges = np.flatnonzero(valid[j])
            next_edges = next_edges[next_edges != i]

            if next_edges.size:
                continuation[i, j] = float(
                    np.max(local_evidence[j, next_edges])
                )

    base_score = (
        0.32 * directed_global
        + 0.28 * local_evidence
        + 0.18 * detour
        + 0.12 * bottleneck
        + 0.10 * continuation
    )

    column_diversity = np.zeros((n, n), dtype=float)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if not indices.size:
            continue

        values = base_score[indices, j]
        positive = values[values > 0.0]

        if positive.size:
            reference = float(np.median(positive))
        else:
            reference = 1.0

        if not np.isfinite(reference) or reference <= 0.0:
            reference = 1.0

        column_diversity[indices, j] = np.clip(
            values / (reference + eps),
            0.0,
            2.0,
        ) / 2.0

    score = 0.90 * base_score + 0.10 * column_diversity
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    return np.clip(score, 0.0, 1.0)
