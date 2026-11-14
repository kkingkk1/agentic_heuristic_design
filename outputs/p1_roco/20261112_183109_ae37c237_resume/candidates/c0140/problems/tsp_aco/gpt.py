import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    valid_directed = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid_directed, False)

    valid = valid_directed | valid_directed.T
    if not np.any(valid):
        return result

    symmetric = np.full((n, n), np.inf, dtype=float)

    both = valid_directed & valid_directed.T
    forward_only = valid_directed & ~valid_directed.T
    backward_only = valid_directed.T & ~valid_directed

    symmetric[both] = 0.5 * (d[both] + d.T[both])
    symmetric[forward_only] = d[forward_only]
    symmetric[backward_only] = d.T[backward_only]
    np.fill_diagonal(symmetric, np.inf)

    eps = np.finfo(float).eps
    finite_values = symmetric[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = max(float(np.median(positive_values)), eps)
    else:
        global_scale = 1.0

    ranks = np.full((n, n), np.inf, dtype=float)
    local_scale = np.full(n, global_scale, dtype=float)
    candidate_count = np.zeros(n, dtype=float)
    regret = np.zeros(n, dtype=float)

    k = min(max(3, int(np.sqrt(n))), n - 1)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        candidate_count[i] = float(neighbors.size)

        if neighbors.size == 0:
            continue

        order = neighbors[
            np.argsort(symmetric[i, neighbors], kind="stable")
        ]
        values = symmetric[i, order]
        ranks[i, order] = np.arange(order.size, dtype=float)

        best = values[0]
        second = values[1] if values.size > 1 else best
        regret[i] = np.clip(
            (second - best) / max(second, eps),
            0.0,
            1.0,
        )

        neighborhood_values = values[: min(k, values.size)]
        positive_local = neighborhood_values[neighborhood_values > 0.0]
        if positive_local.size:
            local_scale[i] = max(
                float(np.median(positive_local)),
                eps,
            )

    row_ratio = np.divide(
        symmetric,
        local_scale[:, None],
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        symmetric,
        local_scale[None, :],
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    global_quality = np.exp(
        -np.clip(symmetric / global_scale, 0.0, 60.0)
    )
    local_quality = np.exp(
        -0.5 * np.clip(row_ratio + col_ratio, 0.0, 60.0)
    )
    rank_quality = 1.0 / (
        1.0 + ranks + ranks.T
    )

    urgency = np.sqrt(
        (1.0 + 0.45 * regret[:, None])
        * (1.0 + 0.45 * regret[None, :])
    )

    nearest_count = min(3, n - 1)
    mutual_nearest = (
        (ranks < nearest_count)
        & (ranks.T < nearest_count)
        & valid
    )
    mutual_factor = 1.0 + 0.14 * mutual_nearest.astype(float)

    candidate_count_safe = np.maximum(candidate_count, 1.0)
    sparsity = np.sqrt(
        (n - 1.0) / candidate_count_safe
    )
    sparsity = np.clip(sparsity, 1.0, np.sqrt(max(n - 1.0, 1.0)))
    sparsity /= max(float(np.max(sparsity)), 1.0)

    scarcity_factor = 1.0 + 0.16 * np.sqrt(
        sparsity[:, None] * sparsity[None, :]
    )

    neighborhood = np.zeros((n, n), dtype=float)
    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size:
            order = neighbors[
                np.argsort(symmetric[i, neighbors], kind="stable")
            ]
            neighborhood[i, order[:k]] = 1.0

    degree = np.maximum(neighborhood.sum(axis=1), 1.0)
    shared = neighborhood @ neighborhood.T
    overlap = shared / np.minimum(degree[:, None], degree[None, :])
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge_factor = 1.0 + 0.18 * (1.0 - overlap)

    raw_score = (
        global_quality ** 0.58
        * local_quality ** 0.24
        * rank_quality ** 0.18
        * urgency
        * mutual_factor
        * scarcity_factor
        * bridge_factor
    )

    raw_score[~valid] = 0.0
    np.fill_diagonal(raw_score, 0.0)
    raw_score = 0.5 * (raw_score + raw_score.T)

    raw_score = np.where(
        np.isfinite(raw_score),
        raw_score,
        0.0,
    )

    total = float(raw_score.sum())
    valid_count = int(np.count_nonzero(valid))

    if total <= 0.0 or valid_count == 0:
        if valid_count:
            result[valid] = 1.0 / valid_count
        return result

    normalized = raw_score / total

    positive_values = normalized[valid]
    concentration = float(np.sum(positive_values * positive_values))
    concentration_ratio = np.clip(
        concentration * valid_count,
        1.0,
        float(valid_count),
    )

    if valid_count > 1:
        concentration_level = (
            concentration_ratio - 1.0
        ) / (valid_count - 1.0)
    else:
        concentration_level = 0.0

    floor_strength = 0.010 + 0.035 * concentration_level
    floor = floor_strength / float(valid_count)

    result = (1.0 - floor_strength) * normalized
    result[valid] += floor
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    final_total = float(result.sum())
    if not np.isfinite(final_total) or final_total <= 0.0:
        result[:] = 0.0
        result[valid] = 1.0 / valid_count
    else:
        result /= final_total

    return result
