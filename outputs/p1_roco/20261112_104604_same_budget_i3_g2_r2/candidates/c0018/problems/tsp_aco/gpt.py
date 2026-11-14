import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    result = np.zeros_like(distance_matrix, dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(distance_matrix)
    valid = finite & (distance_matrix >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    positive_mask = valid & (distance_matrix > 0.0)
    positive = distance_matrix[positive_mask]

    if positive.size:
        q10, q50, q90 = np.percentile(positive, [10.0, 50.0, 90.0])
        base_scale = max(float(q50), eps)
        spread = max(float(q90 - q10), 0.1 * base_scale, eps)
        lower_clip = max(0.25 * float(q10), 1e-3 * base_scale, eps)
        upper_clip = max(float(q90), lower_clip)
    else:
        base_scale = 1.0
        spread = 1.0
        lower_clip = eps
        upper_clip = 1.0

    clipped = np.where(
        valid,
        np.clip(distance_matrix, lower_clip, upper_clip),
        base_scale,
    )

    row_scale = np.full(n, base_scale, dtype=float)
    col_scale = np.full(n, base_scale, dtype=float)
    row_second = np.full(n, base_scale, dtype=float)
    col_second = np.full(n, base_scale, dtype=float)

    for i in range(n):
        outgoing = distance_matrix[i, valid[i]]
        outgoing = outgoing[outgoing > 0.0]
        if outgoing.size:
            clipped_outgoing = np.clip(outgoing, lower_clip, upper_clip)
            row_scale[i] = max(float(np.median(clipped_outgoing)), lower_clip)
            if outgoing.size >= 2:
                row_second[i] = float(np.partition(outgoing, 1)[1])
            else:
                row_second[i] = float(outgoing[0])

        incoming = distance_matrix[valid[:, i], i]
        incoming = incoming[incoming > 0.0]
        if incoming.size:
            clipped_incoming = np.clip(incoming, lower_clip, upper_clip)
            col_scale[i] = max(float(np.median(clipped_incoming)), lower_clip)
            if incoming.size >= 2:
                col_second[i] = float(np.partition(incoming, 1)[1])
            else:
                col_second[i] = float(incoming[0])

    row_scale = np.maximum(row_scale, lower_clip)
    col_scale = np.maximum(col_scale, lower_clip)

    positive_clipped = np.clip(
        np.where(positive_mask, distance_matrix, base_scale),
        lower_clip,
        upper_clip,
    )

    inverse = (upper_clip - positive_clipped) / (
        upper_clip - lower_clip + eps
    )
    inverse = np.where(valid, np.clip(inverse, 0.0, 1.0), 0.0)

    if upper_clip - lower_clip <= 10.0 * eps:
        inverse = np.where(valid, 1.0, 0.0)

    zero_edge = valid & (distance_matrix == 0.0)
    inverse[zero_edge] = 1.0

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

    row_gap = (
        row_second[:, None] - clipped
    ) / (
        row_second[:, None] + clipped + spread + eps
    )
    col_gap = (
        col_second[None, :] - clipped
    ) / (
        col_second[None, :] + clipped + spread + eps
    )
    endpoint_savings = np.minimum(row_gap, col_gap)
    endpoint_savings = np.clip(endpoint_savings, 0.0, 1.0)
    endpoint_savings = np.where(valid, endpoint_savings, 0.0)
    endpoint_savings[zero_edge] = 1.0

    candidate_count = min(
        n - 1,
        max(4, int(np.sqrt(float(n))) + 2),
    )

    outgoing_candidates = []
    incoming_candidates = []

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = np.argsort(
                distance_matrix[i, indices],
                kind="stable",
            )
            outgoing_candidates.append(indices[order[:candidate_count]])
        else:
            outgoing_candidates.append(np.empty(0, dtype=int))

        indices = np.flatnonzero(valid[:, i])
        if indices.size:
            order = np.argsort(
                distance_matrix[indices, i],
                kind="stable",
            )
            incoming_candidates.append(indices[order[:candidate_count]])
        else:
            incoming_candidates.append(np.empty(0, dtype=int))

    two_step_advantage = np.zeros((n, n), dtype=float)

    for i in range(n):
        first_candidates = outgoing_candidates[i]
        if first_candidates.size == 0:
            continue

        for j in range(n):
            if not valid[i, j]:
                continue

            second_candidates = incoming_candidates[j]
            if second_candidates.size == 0:
                continue

            intermediates = np.unique(
                np.concatenate((first_candidates, second_candidates))
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

    score = (
        0.46 * inverse
        + 0.27 * mutual_affinity
        + 0.17 * endpoint_savings
        + 0.10 * two_step_advantage
    )

    score = np.where(valid, score, 0.0)
    maximum = float(np.max(score[valid])) if np.any(valid) else 0.0

    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
