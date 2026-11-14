import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return conservative, structure-aware TSP edge priorities."""

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

    with np.errstate(all="ignore"):
        d = 0.5 * (distances + distances.T)

    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.max(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(finite_values))
    penalty = 5.0 * max(finite_max, global_scale) + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    sorted_work = np.take_along_axis(work, order, axis=1)
    valid_counts = np.sum(valid, axis=1)

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

    relative_distance = np.divide(
        work,
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(work),
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
        np.nan_to_num(
            regret,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )
    regret_bonus = 1.0 + 0.32 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
        & valid
    )
    mutual_bonus = np.where(mutual_nearest, 1.24, 1.0)

    valid_rank = np.where(valid, ranks, float(n))
    rank_fraction = valid_rank / max(1.0, float(n - 1))
    rank_quality = np.clip(1.0 - rank_fraction, 0.0, 1.0)

    candidate_count = np.sum(
        valid & (rank_quality >= 0.5),
        axis=1,
    ).astype(float)

    scarcity = 1.0 / np.sqrt(1.0 + candidate_count)
    scarcity_max = max(float(np.max(scarcity)), eps)
    scarcity /= scarcity_max

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

    neighbor_count = min(4, max(1, n - 1))
    neighbor_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        selected = order[i][valid[i, order[i]]][:neighbor_count]
        neighbor_mask[i, selected] = True

    overlap = (
        neighbor_mask.astype(float)
        @ neighbor_mask.astype(float).T
    )
    neighborhood_sizes = neighbor_mask.sum(axis=1).astype(float)

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
        -0.85 * np.maximum(relative_distance - 1.15, 0.0)
    )

    bridge_bonus = (
        1.0
        + 0.13
        * neighborhood_difference
        * moderate_length_gate
        * quality_gate
    )

    affinity = np.exp(
        -np.clip(work / local_scale[:, None], 0.0, 50.0)
    )
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    node_mass = np.sum(affinity, axis=1)
    probability = np.divide(
        affinity,
        np.maximum(node_mass[:, None], eps),
    )

    affinity_order = np.argsort(
        -affinity,
        axis=1,
        kind="stable",
    )
    top_count = min(2, n)
    top_affinity = np.take_along_axis(
        affinity,
        affinity_order[:, :top_count],
        axis=1,
    )

    best_alternative = top_affinity[:, 0]
    if top_count > 1:
        second_alternative = top_affinity[:, 1]
    else:
        second_alternative = np.zeros(n, dtype=float)

    first_neighbor = affinity_order[:, 0]
    alternative = np.where(
        first_neighbor[:, None] == np.arange(n)[None, :],
        second_alternative[:, None],
        best_alternative[:, None],
    )
    alternative = np.broadcast_to(
        alternative,
        (n, n),
    ).copy()
    alternative[~valid] = 0.0

    alternative_max = float(np.max(alternative))
    if np.isfinite(alternative_max) and alternative_max > eps:
        alternative_normalized = alternative / alternative_max
    else:
        alternative_normalized = np.zeros_like(alternative)

    continuation = np.sqrt(
        np.clip(
            alternative_normalized
            * alternative_normalized.T,
            0.0,
            1.0,
        )
    )

    neighborhood_similarity = np.clip(
        probability @ probability.T,
        0.0,
        1.0,
    )
    cross_region = 1.0 - neighborhood_similarity

    long_term_bonus = (
        1.0
        + quality_gate
        * (
            0.10 * continuation
            + 0.06 * cross_region
        )
    )

    cycle_risk = np.divide(
        overlap,
        float(max(1, neighbor_count)),
    ) * (1.0 - cross_region)

    cycle_penalty = np.exp(
        -0.16 * cycle_risk * quality_gate
    )

    scarcity_bonus = 1.0 + 0.18 * (
        scarcity[:, None] + scarcity[None, :]
    ) * quality_gate

    scores = (
        distance_term
        * (
            0.54
            + 0.27 * bidirectional_rank
            + 0.19 * endpoint_quality
        )
        * regret_bonus
        * mutual_bonus
        * scarcity_bonus
        * bridge_bonus
        * long_term_bonus
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
