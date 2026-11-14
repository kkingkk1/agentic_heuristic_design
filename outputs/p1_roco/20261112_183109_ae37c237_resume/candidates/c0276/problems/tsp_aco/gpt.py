import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded, symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    # Use only mutually valid edges so asymmetric or partially sparse inputs
    # produce a consistent undirected sampling distribution.
    valid = finite & finite.T
    if not np.any(valid):
        return scores

    with np.errstate(all="ignore"):
        d = 0.5 * (distances + distances.T)

    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = float(np.max(finite_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(finite_values))
    invalid_penalty = 6.0 * max(finite_max, global_scale) + global_scale

    work = np.where(valid, d, invalid_penalty)
    np.fill_diagonal(work, invalid_penalty)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_work = np.take_along_axis(work, order, axis=1)
    valid_counts = valid.sum(axis=1).astype(int)

    best = sorted_work[:, 0]
    second = np.where(
        valid_counts >= 2,
        sorted_work[:, 1],
        best,
    )

    best = np.where(np.isfinite(best), best, global_scale)
    second = np.where(np.isfinite(second), second, best)

    local_scale = 0.5 * (best + second)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    relative_distance = np.divide(
        work,
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(work),
    )

    distance_quality = np.exp(
        -np.clip(1.08 * relative_distance, 0.0, 60.0)
    )

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.9 * np.sqrt(float(n)))
    rank_quality = np.exp(
        -0.5 * (
            ranks / rank_scale
            + ranks.T / rank_scale
        )
    )

    endpoint_quality = np.exp(
        -np.clip(
            0.48 * (
                work / local_scale[:, None]
                + work / local_scale[None, :]
            ),
            0.0,
            60.0,
        )
    )

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
    )
    regret = np.clip(
        np.nan_to_num(regret, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    regret_bonus = 1.0 + 0.24 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)

    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
        & valid
    )
    mutual_bonus = np.where(mutual_nearest, 1.16, 1.0)

    candidate_count = np.sum(
        valid & (ranks <= 4.0),
        axis=1,
        dtype=float,
    )
    scarcity = 1.0 / np.sqrt(
        np.maximum(candidate_count, 1.0)
    )
    max_scarcity = max(float(np.max(scarcity)), eps)
    scarcity = scarcity / max_scarcity

    scarcity_bonus = 1.0 + 0.12 * (
        scarcity[:, None] + scarcity[None, :]
    )

    # Finite local neighborhoods.
    neighborhood_size = min(4, n - 1)
    neighborhood = np.zeros((n, n), dtype=bool)

    for i in range(n):
        local_order = order[i]
        selected = local_order[valid[i, local_order]][:neighborhood_size]
        neighborhood[i, selected] = True

    neighborhood_float = neighborhood.astype(float)
    neighborhood_sizes = neighborhood_float.sum(axis=1)

    shared = neighborhood_float @ neighborhood_float.T
    union = (
        neighborhood_sizes[:, None]
        + neighborhood_sizes[None, :]
        - shared
    )

    overlap = np.divide(
        shared,
        np.maximum(np.minimum(
            neighborhood_sizes[:, None],
            neighborhood_sizes[None, :],
        ), 1.0),
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    neighborhood_difference = np.divide(
        union - shared,
        np.maximum(union, 1.0),
    )
    neighborhood_difference = np.clip(
        neighborhood_difference,
        0.0,
        1.0,
    )

    local_affinity = np.exp(
        -np.clip(
            work / local_scale[:, None],
            0.0,
            50.0,
        )
    )
    local_affinity[~valid] = 0.0
    np.fill_diagonal(local_affinity, 0.0)

    candidate_affinity = (
        local_affinity * neighborhood_float
    )

    # Exact best alternatives after excluding the other endpoint.
    alternative = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_candidates = order[i][valid[i, order[i]]]
        if row_candidates.size == 0:
            continue

        first = row_candidates[0]
        first_value = local_affinity[i, first]

        if row_candidates.size > 1:
            second_value = local_affinity[i, row_candidates[1]]
        else:
            second_value = 0.0

        alternative[i, :] = first_value
        alternative[i, first] = second_value

    alternative[~valid] = 0.0

    alternative_scale = max(
        float(np.max(alternative)),
        eps,
    )
    alternative_quality = alternative / alternative_scale

    continuation = np.sqrt(
        np.clip(
            alternative_quality
            * alternative_quality.T,
            0.0,
            1.0,
        )
    )

    local_quality_gate = 0.55 + 0.45 * endpoint_quality

    continuation_bonus = 1.0 + (
        0.13
        * continuation
        * local_quality_gate
    )

    moderate_length = np.exp(
        -0.85 * np.maximum(relative_distance - 1.05, 0.0)
    )

    bridge_gate = (
        moderate_length
        * neighborhood_difference
        * np.sqrt(
            np.clip(
                alternative_quality
                * alternative_quality.T,
                0.0,
                1.0,
            )
        )
        * local_quality_gate
    )

    bridge_bonus = 1.0 + 0.09 * np.clip(
        bridge_gate,
        0.0,
        1.0,
    )

    # Penalize only locally supported edges that close highly overlapping
    # neighborhoods, while preserving a small amount of such connectivity.
    triangle_strength = np.zeros((n, n), dtype=float)
    if n > 2:
        for h in range(n):
            through_h = np.minimum(
                candidate_affinity[:, h][:, None],
                candidate_affinity[h, :][None, :],
            )
            triangle_strength = np.maximum(
                triangle_strength,
                through_h,
            )

    triangle_strength[~valid] = 0.0

    cycle_pressure = np.clip(
        triangle_strength
        * overlap
        * (0.5 + 0.5 * mutual_nearest)
        * local_quality_gate,
        0.0,
        1.0,
    )

    cycle_penalty = 1.0 - 0.07 * cycle_pressure

    scores = (
        distance_quality
        * (
            0.50
            + 0.30 * rank_quality
            + 0.20 * endpoint_quality
        )
        * regret_bonus
        * mutual_bonus
        * scarcity_bonus
        * continuation_bonus
        * bridge_bonus
        * cycle_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
