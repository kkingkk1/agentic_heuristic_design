import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    symmetric_distances = 0.5 * (distances + distances.T)
    symmetric_distances[~valid] = np.inf
    np.fill_diagonal(symmetric_distances, np.inf)

    finite_values = symmetric_distances[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

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
    penalty = max(finite_max, global_scale) * 5.0 + global_scale

    work = np.where(valid, symmetric_distances, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_work = np.take_along_axis(work, order, axis=1)

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    valid_counts = np.sum(valid, axis=1).astype(int)

    first = np.where(valid_counts > 0, sorted_work[:, 0], global_scale)
    second = np.where(
        valid_counts > 1,
        sorted_work[:, 1],
        first,
    )

    first = np.where(np.isfinite(first), first, global_scale)
    second = np.where(np.isfinite(second), second, first)

    local_scale = 0.5 * (first + second)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        work,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=geometric_scale > 0.0,
    )

    distance_term = np.exp(
        -np.clip(1.05 * relative_distance, 0.0, 60.0)
    )

    rank_strength = np.exp(
        -ranks / max(1.0, 0.82 * np.sqrt(float(n)))
    )
    bidirectional_rank = np.sqrt(
        rank_strength * rank_strength.T
    )

    endpoint_quality = np.exp(
        -np.clip(
            0.44
            * (
                work / local_scale[:, None]
                + work / local_scale[None, :]
            ),
            0.0,
            60.0,
        )
    )

    regret = np.divide(
        second - first,
        np.maximum(second, eps),
    )
    regret = np.clip(
        np.nan_to_num(regret, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    regret_bonus = 1.0 + 0.28 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
        & valid
    )
    mutual_bonus = np.where(mutual_nearest, 1.20, 1.0)

    k = min(5, n - 1)
    neighbor_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        candidates = order[i, valid[i, order[i]]]
        neighbor_mask[i, candidates[:k]] = True

    local_affinity = np.exp(
        -np.clip(
            work / np.maximum(local_scale[:, None], eps),
            0.0,
            40.0,
        )
    )
    local_affinity[~valid] = 0.0
    np.fill_diagonal(local_affinity, 0.0)

    candidate_affinity = (
        local_affinity * neighbor_mask.astype(float)
    )

    neighborhood_sizes = neighbor_mask.sum(axis=1).astype(float)
    overlap_count = (
        neighbor_mask.astype(float)
        @ neighbor_mask.astype(float).T
    )
    union_size = (
        neighborhood_sizes[:, None]
        + neighborhood_sizes[None, :]
        - overlap_count
    )
    neighborhood_difference = np.divide(
        union_size - overlap_count,
        np.maximum(union_size, 1.0),
    )
    neighborhood_difference = np.clip(
        neighborhood_difference,
        0.0,
        1.0,
    )

    proximity = np.exp(
        -np.clip(relative_distance, 0.0, 40.0)
    )
    proximity[~valid] = 0.0
    np.fill_diagonal(proximity, 0.0)

    support = np.sum(proximity, axis=1)
    positive_support = support[support > 0.0]
    support_reference = (
        float(np.median(positive_support))
        if positive_support.size
        else 1.0
    )
    support_reference = max(support_reference, eps)

    rarity = np.sqrt(
        support_reference / np.maximum(support, eps)
    )
    rarity = np.clip(rarity, 0.78, 1.45)
    rarity_pair = 0.5 * (
        rarity[:, None] + rarity[None, :]
    )

    alternative = candidate_affinity.copy()
    alternative[~valid] = 0.0

    for i in range(n):
        j = nearest[i]
        if 0 <= j < n:
            alternative[i, j] = 0.0

    alternative_support = np.max(alternative, axis=1)
    alternative_support = np.clip(
        alternative_support,
        0.0,
        1.0,
    )

    moderate_length_gate = np.exp(
        -0.85 * np.maximum(relative_distance - 1.10, 0.0)
    )

    quality_gate = 0.55 + 0.45 * endpoint_quality

    bridge_gate = (
        moderate_length_gate
        * neighborhood_difference
        * quality_gate
        * np.sqrt(
            alternative_support[:, None]
            * alternative_support[None, :]
        )
    )
    bridge_gate = np.clip(bridge_gate, 0.0, 1.0)

    continuation = np.zeros((n, n), dtype=float)

    if n > 2:
        for h in range(n):
            through_h = np.minimum(
                candidate_affinity[:, h][:, None],
                candidate_affinity[h, :][None, :],
            )
            continuation = np.maximum(
                continuation,
                through_h,
            )

    continuation[~valid] = 0.0
    continuation_max = float(np.max(continuation))
    if continuation_max > 0.0 and np.isfinite(continuation_max):
        continuation /= continuation_max

    continuation_bonus = (
        1.0 + 0.10 * continuation * quality_gate
    )

    triangle_strength = continuation.copy()
    closure_pressure = (
        triangle_strength
        * neighborhood_difference
        * (1.0 - 0.45 * bidirectional_rank)
        * quality_gate
    )
    closure_pressure = np.clip(
        closure_pressure,
        0.0,
        1.0,
    )
    cycle_penalty = 1.0 - 0.07 * closure_pressure

    diversity_factor = np.clip(
        1.0
        + 0.045 * (rarity_pair - 1.0)
        + 0.045 * bridge_gate
        - 0.025 * (1.0 - neighborhood_difference),
        0.90,
        1.10,
    )

    scores = (
        distance_term
        * (
            0.54
            + 0.28 * bidirectional_rank
            + 0.18 * endpoint_quality
        )
        * regret_bonus
        * mutual_bonus
        * diversity_factor
        * continuation_bonus
        * (1.0 + 0.10 * bridge_gate)
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
