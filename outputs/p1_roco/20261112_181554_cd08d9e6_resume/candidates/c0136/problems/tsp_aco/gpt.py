import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return conservative, diverse, symmetric TSP edge-priority indicators."""

    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid_directed = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid_directed, False)

    valid = valid_directed | valid_directed.T
    if not np.any(valid):
        return result

    symmetric = np.full((n, n), np.inf, dtype=float)

    both = valid_directed & valid_directed.T
    only_forward = valid_directed & ~valid_directed.T
    only_backward = ~valid_directed & valid_directed.T

    symmetric[both] = 0.5 * (d[both] + d.T[both])
    symmetric[only_forward] = d[only_forward]
    symmetric[only_backward] = d.T[only_backward]
    np.fill_diagonal(symmetric, np.inf)

    finite_values = symmetric[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = 1.0
    else:
        return result

    global_scale = max(global_scale, eps)

    ranks = np.full((n, n), np.inf, dtype=float)
    local_scale = np.full(n, global_scale, dtype=float)
    candidate_count = np.zeros(n, dtype=float)
    regret = np.zeros(n, dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(np.isfinite(symmetric[i]))

        if neighbors.size == 0:
            continue

        ordered = neighbors[
            np.argsort(symmetric[i, neighbors], kind="stable")
        ]
        values = symmetric[i, ordered]

        ranks[i, ordered] = np.arange(ordered.size, dtype=float)

        best = values[0]
        second = values[1] if values.size > 1 else best
        regret[i] = np.clip(
            (second - best) / max(second, eps),
            0.0,
            1.0,
        )

        positive_local = values[: min(4, values.size)]
        positive_local = positive_local[positive_local > 0.0]

        if positive_local.size:
            local_scale[i] = max(
                float(np.median(positive_local)),
                eps,
            )

        candidate_count[i] = float(neighbors.size)

    valid_nodes = candidate_count > 0.0
    max_candidates = max(float(n - 1), 1.0)

    sparsity = np.ones(n, dtype=float)
    sparsity[valid_nodes] = np.sqrt(
        np.clip(
            (max_candidates + 1.0)
            / (candidate_count[valid_nodes] + 1.0),
            1.0,
            max_candidates + 1.0,
        )
    )
    sparsity /= max(float(np.max(sparsity)), 1.0)

    urgency = np.sqrt(
        (1.0 + 0.30 * regret)[:, None]
        * (1.0 + 0.30 * regret)[None, :]
    )

    distance_ratio = np.minimum(symmetric / global_scale, 60.0)
    local_ratio = np.minimum(
        symmetric / np.sqrt(local_scale[:, None] * local_scale[None, :]),
        60.0,
    )

    distance_quality = np.exp(-distance_ratio)
    local_quality = np.exp(-local_ratio)
    rank_quality = 1.0 / (1.0 + ranks + ranks.T)

    nearest_count = min(3, n - 1)
    mutual_nearest = (
        (ranks < nearest_count)
        & (ranks.T < nearest_count)
    ).astype(float)
    mutual_bonus = 1.0 + 0.16 * mutual_nearest

    base_score = (
        distance_quality ** 0.62
        * local_quality ** 0.20
        * rank_quality ** 0.18
        * mutual_bonus
        * urgency
    )

    finite = np.isfinite(symmetric) & valid
    score = np.zeros((n, n), dtype=float)
    score[finite] = base_score[finite]

    endpoint_sparsity = np.sqrt(
        sparsity[:, None] * sparsity[None, :]
    )

    adaptive_floor = (
        0.025
        + 0.055 * endpoint_sparsity
        + 0.020 * np.maximum(
            sparsity[:, None],
            sparsity[None, :],
        )
    ) * distance_quality

    score[finite] = np.maximum(
        score[finite],
        adaptive_floor[finite],
    )

    # Small long-term reserve bonus for edges incident to sparse endpoints.
    reserve_factor = 1.0 + 0.10 * endpoint_sparsity
    score[finite] *= reserve_factor[finite]

    # Build compact neighborhoods for a restrained cross-region bonus.
    neighborhood = np.zeros((n, n), dtype=float)
    neighborhood_size = min(3, n - 1)

    for i in range(n):
        neighbors = np.flatnonzero(finite[i])
        if neighbors.size == 0:
            continue

        ordered = neighbors[
            np.argsort(symmetric[i, neighbors], kind="stable")
        ]
        neighborhood[i, ordered[:neighborhood_size]] = 1.0

    degree = np.maximum(neighborhood.sum(axis=1), 1.0)
    shared = neighborhood @ neighborhood.T
    overlap = shared / np.minimum(degree[:, None], degree[None, :])
    overlap = np.clip(overlap, 0.0, 1.0)

    # Keep bridge support deliberately weaker than distance and rank signals.
    bridge_factor = 1.0 + 0.08 * (1.0 - overlap) * distance_quality
    score[finite] *= bridge_factor[finite]

    # Preserve the exploiter's explicit reserve for each endpoint's best edges.
    reserve_count = min(2, n - 1)

    for i in range(n):
        neighbors = np.flatnonzero(finite[i])
        if neighbors.size == 0:
            continue

        ordered = neighbors[
            np.argsort(symmetric[i, neighbors], kind="stable")
        ]

        for j in ordered[:reserve_count]:
            reserve = (
                0.070
                * distance_quality[i, j]
                * (0.75 + 0.25 * sparsity[i])
            )
            score[i, j] = max(score[i, j], reserve)

    score[~finite] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        result = score / maximum

    result[~np.isfinite(result)] = 0.0
    np.fill_diagonal(result, 0.0)
    return result
