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

    eps = np.finfo(float).eps
    finite = np.isfinite(distance_matrix)
    valid = finite & (distance_matrix >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    positive = distance_matrix[valid & (distance_matrix > 0.0)]

    if positive.size:
        q05, q50, q95 = np.quantile(positive, [0.05, 0.50, 0.95])
        scale = max(float(q50), eps)
        lower = max(float(q05), 1e-3 * scale, eps)
        upper = max(float(q95), lower)
        spread = max(upper - lower, 0.1 * scale, eps)
    else:
        scale = 1.0
        lower = eps
        upper = 1.0
        spread = 1.0

    clipped = np.where(
        valid,
        np.clip(distance_matrix, lower, upper),
        scale,
    )

    zero_edge = valid & (distance_matrix <= eps)

    inverse = (upper - clipped) / (upper - lower + eps)
    inverse = np.where(valid, np.clip(inverse, 0.0, 1.0), 0.0)
    inverse[zero_edge] = 1.0

    row_values = np.where(valid, clipped, np.inf)
    col_values = np.where(valid, clipped, np.inf)

    row_scale = np.full(n, scale, dtype=float)
    col_scale = np.full(n, scale, dtype=float)

    for i in range(n):
        outgoing = clipped[i, valid[i]]
        incoming = clipped[valid[:, i], i]

        if outgoing.size:
            row_scale[i] = max(float(np.median(outgoing)), lower)

        if incoming.size:
            col_scale[i] = max(float(np.median(incoming)), lower)

    row_affinity = np.exp(
        -clipped / (row_scale[:, None] + eps)
    )
    col_affinity = np.exp(
        -clipped / (col_scale[None, :] + eps)
    )

    mutual_affinity = np.sqrt(
        np.maximum(row_affinity, 0.0)
        * np.maximum(col_affinity, 0.0)
    )
    mutual_affinity = np.where(valid, mutual_affinity, 0.0)
    mutual_affinity[zero_edge] = 1.0

    row_sorted = np.sort(row_values, axis=1)
    col_sorted = np.sort(col_values, axis=0)

    row_best = row_sorted[:, 0]
    col_best = col_sorted[0, :]

    row_second = np.where(
        np.isfinite(row_sorted[:, 1]) if n > 1 else False,
        row_sorted[:, 1] if n > 1 else row_best,
        row_best,
    )
    col_second = np.where(
        np.isfinite(col_sorted[1, :]) if n > 1 else False,
        col_sorted[1, :] if n > 1 else col_best,
        col_best,
    )

    row_regret = np.divide(
        np.maximum(row_second - row_best, 0.0),
        row_second + eps,
        out=np.zeros(n, dtype=float),
        where=np.isfinite(row_second),
    )
    col_regret = np.divide(
        np.maximum(col_second - col_best, 0.0),
        col_second + eps,
        out=np.zeros(n, dtype=float),
        where=np.isfinite(col_second),
    )

    row_opportunity = 0.5 + 0.5 * np.clip(row_regret, 0.0, 1.0)
    col_opportunity = 0.5 + 0.5 * np.clip(col_regret, 0.0, 1.0)

    endpoint_scarcity = np.sqrt(
        row_opportunity[:, None] * col_opportunity[None, :]
    )
    endpoint_scarcity = np.where(valid, endpoint_scarcity, 0.0)

    row_rank = np.full((n, n), n + 1.0, dtype=float)
    col_rank = np.full((n, n), n + 1.0, dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[
                np.argsort(distance_matrix[i, indices], kind="stable")
            ]
            row_rank[i, order] = np.arange(1, order.size + 1)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = indices[
                np.argsort(distance_matrix[indices, j], kind="stable")
            ]
            col_rank[order, j] = np.arange(1, order.size + 1)

    row_count = np.sum(valid, axis=1, keepdims=True)
    col_count = np.sum(valid, axis=0, keepdims=True)

    outgoing_quality = np.where(
        valid,
        1.0 - (row_rank - 1.0) / np.maximum(row_count, 1.0),
        0.0,
    )
    incoming_quality = np.where(
        valid,
        1.0 - (col_rank - 1.0) / np.maximum(col_count, 1.0),
        0.0,
    )

    endpoint_agreement = np.sqrt(
        np.maximum(outgoing_quality, 0.0)
        * np.maximum(incoming_quality, 0.0)
    )

    local_signal = (
        0.55 * mutual_affinity
        + 0.25 * outgoing_quality
        + 0.20 * endpoint_agreement
    )
    local_signal = np.where(valid, local_signal, 0.0)

    global_closeness = np.where(
        valid,
        np.clip(
            (upper - clipped) / (upper - lower + eps),
            0.0,
            1.0,
        ),
        0.0,
    )
    global_closeness[zero_edge] = 1.0

    row_mass = np.sum(global_closeness, axis=1, keepdims=True)
    col_mass = np.sum(global_closeness, axis=0, keepdims=True)

    row_share = np.divide(
        global_closeness,
        row_mass,
        out=np.zeros_like(global_closeness),
        where=row_mass > eps,
    )
    col_share = np.divide(
        global_closeness,
        col_mass,
        out=np.zeros_like(global_closeness),
        where=col_mass > eps,
    )

    row_dominance = np.minimum(n * row_share, 1.0)
    col_dominance = np.minimum(n * col_share, 1.0)

    diversity_reserve = 1.0 - np.sqrt(
        row_dominance * col_dominance
    )
    diversity_reserve = np.where(valid, diversity_reserve, 0.0)

    global_signal = np.sqrt(
        np.maximum(global_closeness, 0.0)
        * (0.45 + 0.55 * diversity_reserve)
    )
    global_signal = np.where(valid, global_signal, 0.0)

    two_step_advantage = np.zeros((n, n), dtype=float)
    candidate_count = min(
        n - 1,
        max(4, int(np.sqrt(float(n))) + 2),
    )

    outgoing_candidates = []
    incoming_candidates = []

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[
                np.argsort(distance_matrix[i, indices], kind="stable")
            ]
            outgoing_candidates.append(order[:candidate_count])
        else:
            outgoing_candidates.append(np.empty(0, dtype=int))

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = indices[
                np.argsort(distance_matrix[indices, j], kind="stable")
            ]
            incoming_candidates.append(order[:candidate_count])
        else:
            incoming_candidates.append(np.empty(0, dtype=int))

    for i in range(n):
        for j in range(n):
            if not valid[i, j]:
                continue

            intermediates = np.unique(
                np.concatenate(
                    (outgoing_candidates[i], incoming_candidates[j])
                )
            )
            intermediates = intermediates[
                (intermediates != i) & (intermediates != j)
            ]

            if intermediates.size == 0:
                continue

            usable = valid[i, intermediates] & valid[intermediates, j]
            if not np.any(usable):
                continue

            intermediates = intermediates[usable]
            alternative = np.min(
                clipped[i, intermediates]
                + clipped[intermediates, j]
            )
            direct = clipped[i, j]

            advantage = (alternative - direct) / (
                alternative + direct + spread + eps
            )
            two_step_advantage[i, j] = np.clip(advantage, 0.0, 1.0)

    two_step_advantage[zero_edge] = 1.0

    long_term_signal = (
        0.50 * endpoint_scarcity
        + 0.25 * global_signal
        + 0.15 * two_step_advantage
        + 0.10 * endpoint_agreement
    )
    long_term_signal = np.where(valid, long_term_signal, 0.0)

    score = (
        0.38 * inverse
        + 0.27 * local_signal
        + 0.20 * long_term_signal
        + 0.10 * global_signal
        + 0.05 * two_step_advantage
    )

    score = np.where(valid, np.maximum(score, 0.0), 0.0)

    maximum = float(np.max(score[valid])) if np.any(valid) else 0.0
    if maximum > 0.0:
        score /= maximum

    score[zero_edge] = 1.0
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
