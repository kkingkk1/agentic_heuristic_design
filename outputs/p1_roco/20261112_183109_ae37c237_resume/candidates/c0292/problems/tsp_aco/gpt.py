import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded, symmetric edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)
    valid = finite & finite.T

    if not np.any(valid):
        return result

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    positive = d[valid]
    positive = positive[positive > 0.0]

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = d[valid]
        global_scale = (
            float(np.max(finite_values))
            if finite_values.size
            else 1.0
        )

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(d[valid]))
    fallback = max(finite_max, global_scale) * 8.0 + global_scale

    work = np.where(valid, d, fallback)
    np.fill_diagonal(work, fallback)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    nearest_values = np.take_along_axis(work, order[:, :3], axis=1)
    best = nearest_values[:, 0]

    if n >= 3:
        second = nearest_values[:, 1]
        third = nearest_values[:, 2]
    else:
        second = best
        third = second

    local_scale = 0.5 * (best + second)
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

    relative_i = np.divide(
        work,
        local_scale[:, None],
        out=np.full((n, n), np.inf, dtype=float),
        where=local_scale[:, None] > 0.0,
    )
    relative_j = np.divide(
        work,
        local_scale[None, :],
        out=np.full((n, n), np.inf, dtype=float),
        where=local_scale[None, :] > 0.0,
    )

    rank_decay = max(1.0, 0.72 * np.sqrt(float(n)))
    rank_strength = np.exp(
        -np.clip(ranks / rank_decay, 0.0, 40.0)
    )
    bidirectional_rank = np.sqrt(
        rank_strength * rank_strength.T
    )

    distance_term = np.exp(
        -np.clip(1.12 * relative_distance, 0.0, 60.0)
    )

    endpoint_term = np.exp(
        -np.clip(
            0.46 * (relative_i + relative_j),
            0.0,
            60.0,
        )
    )

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
    )
    regret = np.clip(regret, 0.0, 1.0)

    local_regret = np.divide(
        third - second,
        np.maximum(third, eps),
    )
    local_regret = np.clip(local_regret, 0.0, 1.0)

    regret_bonus = 1.0 + 0.28 * (
        regret[:, None] + regret[None, :]
    )
    regret_bonus *= 1.0 + 0.10 * (
        local_regret[:, None] + local_regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)

    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.34, 1.0)

    # Candidate support estimates how many locally acceptable alternatives
    # remain available for each endpoint.
    support_limit = 1.85 * local_scale[:, None]
    support = (
        (work <= support_limit)
        & valid
    )
    np.fill_diagonal(support, False)

    support_count = np.sum(support, axis=1).astype(float)
    positive_support = support_count[support_count > 0.0]

    if positive_support.size:
        support_reference = float(np.median(positive_support))
    else:
        support_reference = 1.0

    endpoint_pressure = np.sqrt(
        support_reference
        / np.maximum(support_count, 1.0)
    )
    endpoint_pressure = np.clip(endpoint_pressure, 0.84, 1.22)

    feasibility_term = np.sqrt(
        endpoint_pressure[:, None]
        * endpoint_pressure[None, :]
    )
    feasibility_term = np.clip(feasibility_term, 0.88, 1.15)

    # Neighborhood mass distinguishes isolated nodes from crowded nodes.
    neighborhood_kernel = np.exp(
        -np.clip(relative_i, 0.0, 24.0)
    )
    neighborhood_kernel[~valid] = 0.0
    np.fill_diagonal(neighborhood_kernel, 0.0)

    neighborhood_mass = np.sum(neighborhood_kernel, axis=1)
    positive_mass = neighborhood_mass[neighborhood_mass > 0.0]

    if positive_mass.size:
        mass_reference = float(np.median(positive_mass))
    else:
        mass_reference = 1.0

    scarcity = np.sqrt(
        mass_reference
        / np.maximum(neighborhood_mass, eps)
    )
    scarcity = np.clip(scarcity, 0.80, 1.25)

    scarcity_term = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )
    scarcity_term = np.clip(scarcity_term, 0.87, 1.16)

    # Penalize edges whose endpoints have highly overlapping local choices.
    # Such edges are more likely to remain inside one dense cluster.
    close_i = (
        work <= 1.60 * local_scale[:, None]
    ) & valid
    close_j = (
        work <= 1.60 * local_scale[None, :]
    ) & valid

    shared_neighbors = (
        close_i.astype(float) @ close_j.astype(float)
    )

    degree_i = np.sum(close_i, axis=1).astype(float)
    degree_j = np.sum(close_j, axis=0).astype(float)

    overlap_denominator = np.maximum(
        0.5 * (degree_i[:, None] + degree_j[None, :]),
        1.0,
    )
    overlap = shared_neighbors / overlap_denominator
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge_bonus = 1.0 + 0.08 * (1.0 - overlap)
    bridge_bonus = np.where(
        relative_distance <= 2.5,
        bridge_bonus,
        1.0 + 0.08 * (1.0 - overlap)
        * np.exp(-0.65 * (relative_distance - 2.5)),
    )

    long_edge_penalty = np.exp(
        -0.82 * np.maximum(relative_distance - 1.0, 0.0)
    )

    # Combine factors in log space for stable relative ranking.
    safe_terms = [
        distance_term,
        0.56 + 0.28 * bidirectional_rank + 0.16 * endpoint_term,
        regret_bonus,
        mutual_bonus,
        scarcity_term,
        feasibility_term,
        bridge_bonus,
        long_edge_penalty,
    ]

    log_score = np.zeros((n, n), dtype=float)
    for term in safe_terms:
        log_score += np.log(np.maximum(term, eps))

    log_score[~valid] = -np.inf
    finite_log = log_score[np.isfinite(log_score)]

    if finite_log.size:
        reference = float(np.max(finite_log))
        result = np.exp(
            np.clip(log_score - reference, -60.0, 0.0)
        )
    else:
        result.fill(0.0)

    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
