import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a finite, nonnegative, symmetric edge-prior matrix for TSP."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(raw) & (raw > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    # Build a symmetric distance matrix while retaining one-sided valid edges.
    transpose_valid = valid.T
    both_valid = valid & transpose_valid
    only_forward = valid & ~transpose_valid
    only_backward = transpose_valid & ~valid

    d = np.full((n, n), np.inf, dtype=float)
    d[both_valid] = 0.5 * (raw[both_valid] + raw.T[both_valid])
    d[only_forward] = raw[only_forward]
    d[only_backward] = raw.T[only_backward]
    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    positive = d[finite & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    scale = max(scale, np.finfo(float).eps)

    # Robust inverse-distance preference.
    inverse = np.zeros_like(d)
    inverse[finite] = scale / (d[finite] + 0.05 * scale)

    # Endpoint-local distance scales from the nearest few valid neighbors.
    local_scale = np.full(n, scale, dtype=float)
    row_medians = np.full(n, scale, dtype=float)

    for i in range(n):
        row_values = d[i, finite[i]]
        if row_values.size:
            k = min(5, row_values.size)
            nearest_values = np.partition(row_values, k - 1)[:k]
            local_scale[i] = max(
                float(np.median(nearest_values)),
                np.finfo(float).eps,
            )
            row_medians[i] = max(
                float(np.median(row_values)),
                np.finfo(float).eps,
            )

    pair_local_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = d / np.maximum(pair_local_scale, np.finfo(float).eps)
    proximity = np.exp(-np.minimum(relative_distance, 60.0))
    proximity[~finite] = 0.0

    # Rank-based preference at each endpoint.
    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.full((n, n), n + 1.0, dtype=float)

    for i in range(n):
        valid_order = order[i, np.isfinite(d[i, order[i]])]
        ranks[i, valid_order] = np.arange(1, valid_order.size + 1)

    rank_scale = max(1.0, n / 4.0)
    rank_preference = np.exp(-(ranks - 1.0) / rank_scale)
    rank_preference[~finite] = 0.0
    bidirectional_rank = np.sqrt(
        rank_preference * rank_preference.T
    )

    # Row-normalized preference prevents a few hubs from dominating.
    row_sum = inverse.sum(axis=1, keepdims=True)
    local_preference = np.divide(
        inverse,
        row_sum,
        out=np.zeros_like(inverse),
        where=row_sum > 0,
    )
    endpoint_preference = np.sqrt(
        local_preference * local_preference.T
    )

    # Mutual-nearest-neighbor evidence.
    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        if np.any(finite[i]):
            nearest[i] = int(np.argmin(d[i]))

    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    # Softly suppress edges that are unusually long for either endpoint.
    excess_i = np.maximum(
        0.0,
        d - row_medians[:, None],
    ) / scale
    excess_j = np.maximum(
        0.0,
        d - row_medians[None, :],
    ) / scale
    long_edge_gate = 1.0 / (1.0 + 0.5 * (excess_i + excess_j))
    long_edge_gate[~finite] = 0.0

    score = (
        0.36 * proximity
        + 0.24 * endpoint_preference
        + 0.20 * bidirectional_rank
        + 0.12 * local_preference
        + 0.08 * mutual
    )
    score *= 0.65 + 0.35 * long_edge_gate

    score = 0.5 * (score + score.T)
    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
