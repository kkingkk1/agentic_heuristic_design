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

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.max(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(finite_values))
    penalty = max(finite_max, global_scale) * 5.0 + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_work = np.take_along_axis(work, order, axis=1)

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    valid_counts = np.sum(valid, axis=1).astype(int)

    first = np.where(valid_counts > 0, sorted_work[:, 0], global_scale)
    second = np.where(valid_counts > 1, sorted_work[:, 1], first)

    local_scale = 0.5 * (first + second)
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
        where=(local_scale[:, None] * local_scale[None, :]) > 0.0,
    )

    distance_term = np.exp(
        -np.clip(1.05 * relative_distance, 0.0, 60.0)
    )

    rank_strength = np.exp(
        -ranks / max(1.0, 0.80 * np.sqrt(float(n)))
    )
    bidirectional_rank = np.sqrt(
        rank_strength * rank_strength.T
    )

    endpoint_quality = np.exp(
        -np.clip(
            0.42
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
    regret_bonus = 1.0 + 0.30 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)

    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
        & valid
    )
    mutual_bonus = np.where(mutual_nearest, 1.22, 1.0)

    valid_rank = np.where(valid, ranks, float(n))
    rank_fraction = valid_rank / max(1.0, float(n - 1))
    rank_quality = np.clip(1.0 - rank_fraction, 0.0, 1.0)

    candidate_count = np.sum(
        valid & (rank_quality >= 0.5),
        axis=1,
    ).astype(float)

    scarcity = 1.0 / np.sqrt(1.0 + candidate_count)
    max_scarcity = max(float(np.max(scarcity)), eps)
    scarcity = scarcity / max_scarcity

    local_quality = np.exp(
        -0.48
        * np.clip(
            work / local_scale[:, None]
            + work / local_scale[None, :],
            0.0,
            60.0,
        )
    )
    quality_gate = 0.55 + 0.45 * local_quality

    k = min(5, max(1, n - 1))
    neighbor_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        row_order = order[i]
        selected = row_order[valid[i, row_order]][:k]
        neighbor_mask[i, selected] = True

    neighborhood_sizes = neighbor_mask.sum(axis=1).astype(float)
    overlap = neighbor_mask.astype(float) @ neighbor_mask.astype(float).T

    union_size = (
        neighborhood_sizes[:, None]
        + neighborhood_sizes[None, :]
        - overlap
    )
    neighborhood_difference = np.divide(
        union_size - overlap,
        np.maximum(union_size, 1.0),
    )
    neighborhood_difference = np.clip(
        neighborhood_difference,
        0.0,
        1.0,
    )

    moderate_length_gate = np.exp(
        -0.90 * np.maximum(relative_distance - 1.10, 0.0)
    )

    local_affinity = np.exp(
        -np.clip(
            work / np.maximum(local_scale[:, None], eps),
            0.0,
            40.0,
        )
    )
    local_affinity[~valid] = 0.0
    np.fill_diagonal(local_affinity, 0.0)

    candidate_affinity = local_affinity * neighbor_mask.astype(float)

    continuation = np.zeros((n, n), dtype=float)
    if n > 2:
        for h in range(n):
            through_h = np.minimum(
                candidate_affinity[:, h][:, None],
                candidate_affinity[h, :][None, :],
            )
            continuation = np.maximum(continuation, through_h)

    continuation[~valid] = 0.0
    continuation_max = max(float(np.max(continuation)), eps)
    continuation = continuation / continuation_max

    continuation_bonus = (
        1.0
        + 0.12 * continuation * quality_gate
    )

    alternative = candidate_affinity.copy()
    alternative[~valid] = 0.0

    for i in range(n):
        j = nearest[i]
        if 0 <= j < n:
            alternative[i, j] = 0.0

    alternative_support = np.max(alternative, axis=1)
    alternative_support = np.clip(alternative_support, 0.0, 1.0)

    bridge_gate = (
        moderate_length_gate
        * neighborhood_difference
        * quality_gate
        * np.sqrt(
            alternative_support[:, None]
            * alternative_support[None, :]
        )
    )

    bridge_bonus = 1.0 + 0.11 * bridge_gate

    triangle_strength = np.zeros((n, n), dtype=float)
    if n > 2:
        for h in range(n):
            triangle_strength = np.maximum(
                triangle_strength,
                np.minimum(
                    candidate_affinity[:, h][:, None],
                    candidate_affinity[h, :][None, :],
                ),
            )

    triangle_strength[~valid] = 0.0

    closure_pressure = (
        triangle_strength
        * neighborhood_difference
        * (1.0 - 0.5 * bidirectional_rank)
        * quality_gate
    )
    closure_pressure = np.clip(closure_pressure, 0.0, 1.0)

    cycle_penalty = 1.0 - 0.08 * closure_pressure

    scarcity_bonus = 1.0 + 0.16 * (
        scarcity[:, None] + scarcity[None, :]
    ) * quality_gate

    scores = (
        distance_term
        * (
            0.55
            + 0.27 * bidirectional_rank
            + 0.18 * endpoint_quality
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

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
