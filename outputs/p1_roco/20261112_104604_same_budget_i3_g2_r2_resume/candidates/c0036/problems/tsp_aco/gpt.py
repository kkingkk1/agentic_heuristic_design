import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = (
        np.isfinite(distance_matrix)
        & (distance_matrix >= 0.0)
        & off_diagonal
    )

    if not np.any(valid):
        return result

    distances = np.where(valid, distance_matrix, np.nan)
    valid_values = distance_matrix[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.mean(positive_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps

    global_affinity = np.zeros((n, n), dtype=float)
    global_affinity[valid] = np.exp(
        -distance_matrix[valid] / (scale + eps)
    )

    row_affinity = np.zeros((n, n), dtype=float)
    row_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue

        values = distance_matrix[i, indices]
        positive_local = values[values > 0.0]

        if positive_local.size:
            local_scale = float(np.median(positive_local))
        else:
            local_scale = scale

        if not np.isfinite(local_scale) or local_scale <= 0.0:
            local_scale = scale

        row_affinity[i, indices] = np.exp(
            -values / (local_scale + eps)
        )

        order = indices[np.argsort(values, kind="stable")]
        for rank, j in enumerate(order):
            row_rank[i, j] = 1.0 / (1.0 + float(rank))

    incoming_affinity = np.zeros((n, n), dtype=float)
    incoming_rank = np.zeros((n, n), dtype=float)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size == 0:
            continue

        values = distance_matrix[indices, j]
        positive_local = values[values > 0.0]

        if positive_local.size:
            local_scale = float(np.median(positive_local))
        else:
            local_scale = scale

        if not np.isfinite(local_scale) or local_scale <= 0.0:
            local_scale = scale

        incoming_affinity[indices, j] = np.exp(
            -values / (local_scale + eps)
        )

        order = indices[np.argsort(values, kind="stable")]
        for rank, i in enumerate(order):
            incoming_rank[i, j] = 1.0 / (1.0 + float(rank))

    local_affinity = np.sqrt(
        np.maximum(row_affinity, 0.0)
        * np.maximum(incoming_affinity, 0.0)
    )

    rank_signal = np.sqrt(
        np.maximum(row_rank, 0.0)
        * np.maximum(incoming_rank, 0.0)
    )

    regret = np.zeros((n, n), dtype=float)

    for i in range(n):
        outgoing = np.flatnonzero(valid[i])
        for j in outgoing:
            alternatives_out = outgoing[outgoing != j]
            if alternatives_out.size:
                best_out = float(np.min(distance_matrix[i, alternatives_out]))
                source_gain = max(
                    best_out - float(distance_matrix[i, j]), 0.0
                ) / (best_out + float(distance_matrix[i, j]) + eps)
            else:
                source_gain = 0.0

            incoming = np.flatnonzero(valid[:, j])
            alternatives_in = incoming[incoming != i]
            if alternatives_in.size:
                best_in = float(np.min(distance_matrix[alternatives_in, j]))
                target_gain = max(
                    best_in - float(distance_matrix[i, j]), 0.0
                ) / (best_in + float(distance_matrix[i, j]) + eps)
            else:
                target_gain = 0.0

            regret[i, j] = 0.5 * (source_gain + target_gain)

    score = (
        0.15 * global_affinity
        + 0.35 * local_affinity
        + 0.25 * rank_signal
        + 0.25 * regret
    )

    score[~valid] = 0.0

    for i in range(n):
        row = score[i]
        maximum = float(np.max(row))
        if maximum > 0.0 and np.isfinite(maximum):
            score[i] = row / maximum

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    return score
