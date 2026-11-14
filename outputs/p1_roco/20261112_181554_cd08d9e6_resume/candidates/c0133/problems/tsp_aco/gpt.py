import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally normalized symmetric edge-priority indicators for TSP."""

    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    directed_valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(directed_valid, False)

    valid = directed_valid | directed_valid.T
    if not np.any(valid):
        return result

    symmetric = np.full((n, n), np.inf, dtype=float)

    both = directed_valid & directed_valid.T
    forward_only = directed_valid & ~directed_valid.T
    backward_only = ~directed_valid & directed_valid.T

    symmetric[both] = 0.5 * (d[both] + d.T[both])
    symmetric[forward_only] = d[forward_only]
    symmetric[backward_only] = d.T[backward_only]
    np.fill_diagonal(symmetric, np.inf)

    finite_values = symmetric[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = max(float(np.median(positive_values)), eps)
    elif finite_values.size:
        global_scale = 1.0
    else:
        return result

    local_scale = np.full(n, global_scale, dtype=float)
    best = np.full(n, np.inf, dtype=float)
    second = np.full(n, np.inf, dtype=float)
    ranks = np.full((n, n), np.inf, dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        order = neighbors[
            np.argsort(symmetric[i, neighbors], kind="stable")
        ]
        values = symmetric[i, order]

        ranks[i, order] = np.arange(order.size, dtype=float)
        best[i] = values[0]
        second[i] = values[1] if values.size > 1 else values[0]

        local_values = values[: min(3, values.size)]
        positive_local = local_values[local_values > 0.0]
        if positive_local.size:
            local_scale[i] = max(
                float(np.median(positive_local)),
                eps,
            )

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )

    safe_second = np.where(
        np.isfinite(second),
        second,
        np.maximum(best, local_scale),
    )
    safe_best = np.where(
        np.isfinite(best),
        best,
        local_scale,
    )

    regret = np.divide(
        safe_second - safe_best,
        np.maximum(safe_second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(safe_second),
    )
    regret = np.clip(regret, 0.0, 1.0)

    endpoint_urgency = np.sqrt(
        (1.0 + 0.42 * regret)[:, None]
        * (1.0 + 0.42 * regret)[None, :]
    )

    distance_quality = np.exp(
        -np.clip(symmetric / global_scale, 0.0, 60.0)
    )

    local_denominator = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    local_quality = np.exp(
        -np.clip(symmetric / local_denominator, 0.0, 60.0)
    )

    rank_quality = 1.0 / (
        1.0 + ranks + ranks.T
    )

    nearest_count = min(3, n - 1)
    mutual_nearest = (
        (ranks < nearest_count)
        & (ranks.T < nearest_count)
    )
    mutual_bonus = 1.0 + 0.16 * mutual_nearest.astype(float)

    # Exploitation-dominant base with restrained local corrections.
    base_score = (
        distance_quality ** 0.62
        * local_quality ** 0.23
        * rank_quality ** 0.15
        * endpoint_urgency
        * mutual_bonus
    )

    finite = valid & np.isfinite(symmetric)
    score = np.zeros((n, n), dtype=float)
    score[finite] = base_score[finite]

    # Preserve a small global bridge floor without allowing long edges to
    # dominate the distance-based signal.
    bridge_floor = 0.028 * distance_quality
    score[finite] = np.maximum(
        score[finite],
        bridge_floor[finite],
    )

    # Retain a small amount of endpoint reserve for sparse or high-regret
    # nodes.
    reserve_count = min(2, n - 1)
    for i in range(n):
        candidates = np.flatnonzero(finite[i])
        if candidates.size == 0:
            continue

        ordered = candidates[
            np.argsort(symmetric[i, candidates], kind="stable")
        ]
        for j in ordered[:reserve_count]:
            reserve = (
                0.055
                * distance_quality[i, j]
                * (1.0 + 0.25 * regret[i])
            )
            score[i, j] = max(score[i, j], reserve)

    # Add a restrained Explorer-style bridge signal based on compact
    # neighborhoods and their overlap.
    k = min(max(3, int(np.sqrt(n))), n - 1)
    neighborhood = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(finite[i])
        if candidates.size:
            ordered = candidates[
                np.argsort(symmetric[i, candidates], kind="stable")
            ]
            neighborhood[i, ordered[:k]] = 1.0

    degree = np.maximum(neighborhood.sum(axis=1), 1.0)
    overlap = (neighborhood @ neighborhood.T) / np.minimum(
        degree[:, None],
        degree[None, :],
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge_factor = 1.0 + 0.14 * (1.0 - overlap)

    reach = neighborhood @ neighborhood.T
    reach_scale = np.maximum(reach.max(axis=1), 1.0)
    access = np.divide(
        reach.sum(axis=1),
        reach_scale,
        out=np.zeros(n, dtype=float),
        where=reach_scale > 0.0,
    )
    access_factor = 1.0 + 0.05 * (
        1.0 - np.sqrt(access[:, None] * access[None, :])
    )

    score[finite] *= (
        bridge_factor[finite]
        * access_factor[finite]
    )

    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    # Enforce symmetry after all endpoint-specific adjustments.
    score = 0.5 * (score + score.T)

    total = float(score.sum())
    if not np.isfinite(total) or total <= 0.0:
        count = float(np.count_nonzero(finite))
        if count > 0.0:
            result[finite] = 1.0 / count
        return result

    normalized = score / total

    # Mild adaptive exploration floor: only concentrated distributions
    # receive a noticeable reserve for less competitive valid edges.
    positive = normalized[finite]
    valid_count = float(positive.size)
    concentration = float(np.sum(positive * positive))
    concentration_ratio = np.clip(
        concentration * valid_count,
        1.0,
        valid_count,
    )

    floor_strength = 0.010 + 0.014 * (
        (concentration_ratio - 1.0)
        / max(valid_count - 1.0, 1.0)
    )
    floor = floor_strength / valid_count

    result = (1.0 - floor_strength) * normalized
    result[finite] += floor
    result[~finite] = 0.0
    np.fill_diagonal(result, 0.0)

    return result
