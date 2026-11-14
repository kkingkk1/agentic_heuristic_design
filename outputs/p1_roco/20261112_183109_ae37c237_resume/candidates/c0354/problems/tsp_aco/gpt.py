import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded, symmetric edge-priority indicators for TSP."""
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
    elif finite_values.size:
        global_scale = float(np.max(finite_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(finite_values))
    penalty = max(finite_max, global_scale) * 6.0 + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    sorted_distances = np.sort(work, axis=1)
    best = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 2 else best

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

    distance_term = np.exp(
        -np.clip(1.24 * relative_distance, 0.0, 60.0)
    )

    rank_strength = np.exp(
        -ranks / max(1.0, 0.72 * np.sqrt(float(n)))
    )
    bidirectional_rank = np.sqrt(
        rank_strength * rank_strength.T
    )

    endpoint_quality = np.exp(
        -np.clip(
            0.40
            * (
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
    regret = np.clip(regret, 0.0, 1.0)

    regret_bonus = 1.0 + 0.36 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.32, 1.0)

    long_edge_penalty = np.exp(
        -0.88 * np.maximum(relative_distance - 1.0, 0.0)
    )

    neighborhood_kernel = np.exp(
        -np.clip(work / local_scale[:, None], 0.0, 24.0)
    )
    neighborhood_kernel[~valid] = 0.0
    np.fill_diagonal(neighborhood_kernel, 0.0)

    neighborhood_mass = np.sum(neighborhood_kernel, axis=1)
    positive_mass = neighborhood_mass[neighborhood_mass > 0.0]
    mass_reference = (
        float(np.median(positive_mass))
        if positive_mass.size
        else 1.0
    )

    if not np.isfinite(mass_reference) or mass_reference <= 0.0:
        mass_reference = 1.0

    scarcity = np.sqrt(
        mass_reference
        / np.maximum(neighborhood_mass, eps)
    )
    scarcity = np.clip(scarcity, 0.80, 1.22)

    scarcity_term = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )
    scarcity_term = np.clip(scarcity_term, 0.89, 1.12)

    support_kernel = np.exp(
        -np.clip(work / local_scale[:, None], 0.0, 18.0)
    )
    support_kernel[~valid] = 0.0
    np.fill_diagonal(support_kernel, 0.0)

    support_count = np.sum(
        support_kernel >= np.exp(-1.60),
        axis=1,
    ).astype(float)

    positive_support = support_count[support_count > 0.0]
    support_reference = (
        float(np.median(positive_support))
        if positive_support.size
        else 1.0
    )

    if not np.isfinite(support_reference) or support_reference <= 0.0:
        support_reference = 1.0

    endpoint_need = np.sqrt(
        support_reference
        / np.maximum(support_count, 1.0)
    )
    endpoint_need = np.clip(endpoint_need, 0.88, 1.15)

    feasibility_term = np.sqrt(
        endpoint_need[:, None] * endpoint_need[None, :]
    )
    feasibility_term = np.clip(feasibility_term, 0.92, 1.10)

    close_to_i = (
        work <= 1.45 * local_scale[:, None]
    )
    close_to_j = (
        work <= 1.45 * local_scale[None, :]
    )
    close_to_i[~valid] = False
    close_to_j[~valid] = False

    shared_neighbors = close_to_i.astype(float) @ close_to_j.astype(float)
    endpoint_degree = (
        np.sum(close_to_i, axis=1)
        + np.sum(close_to_j, axis=0)
    )

    overlap = np.divide(
        shared_neighbors,
        np.maximum(0.5 * endpoint_degree, 1.0),
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge_term = 1.0 + 0.035 * (1.0 - overlap)
    bridge_term = np.clip(bridge_term, 1.0, 1.035)

    scores = (
        distance_term
        * (
            0.54
            + 0.30 * bidirectional_rank
            + 0.16 * endpoint_quality
        )
        * regret_bonus
        * mutual_bonus
        * scarcity_term
        * feasibility_term
        * bridge_term
        * long_edge_penalty
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
