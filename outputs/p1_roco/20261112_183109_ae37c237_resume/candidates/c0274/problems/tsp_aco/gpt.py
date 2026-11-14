import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, scale-invariant edge-priority indicators for TSP."""
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

    eps = np.finfo(float).eps
    valid_values = d[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.max(valid_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(valid_values))
    penalty = max(finite_max, global_scale) * 6.0 + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    sorted_distances = np.take_along_axis(work, order, axis=1)
    degree = np.sum(valid, axis=1)

    best = sorted_distances[:, 0]
    second = (
        sorted_distances[:, 1]
        if n > 2
        else best.copy()
    )
    second = np.where(degree >= 2, second, best)

    best = np.where(np.isfinite(best), best, global_scale)
    second = np.where(np.isfinite(second), second, best)

    neighborhood_size = min(3, n - 1)
    neighbor_block = sorted_distances[:, :neighborhood_size]
    neighbor_valid = (
        np.arange(neighborhood_size)[None, :]
        < np.minimum(degree[:, None], neighborhood_size)
    )

    neighbor_sum = np.sum(
        np.where(neighbor_valid, neighbor_block, 0.0),
        axis=1,
    )
    neighbor_count = np.maximum(
        np.sum(neighbor_valid, axis=1),
        1.0,
    )

    local_scale = neighbor_sum / neighbor_count
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
        -np.clip(1.08 * relative_distance, 0.0, 60.0)
    )

    rank_strength = np.exp(
        -ranks / max(1.0, 0.90 * np.sqrt(float(n)))
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
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second),
    )
    regret = np.clip(regret, 0.0, 1.0)

    regret_term = 1.0 + 0.32 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)
    has_neighbor = degree > 0

    mutual_nearest = (
        has_neighbor[:, None]
        & has_neighbor[None, :]
        & (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_term = np.where(mutual_nearest, 1.28, 1.0)

    long_edge_term = np.exp(
        -0.62 * np.maximum(relative_distance - 1.0, 0.0)
    )

    proximity = np.exp(
        -np.clip(relative_distance, 0.0, 30.0)
    )
    proximity[~valid] = 0.0
    np.fill_diagonal(proximity, 0.0)

    neighborhood_mass = np.sum(proximity, axis=1)
    mass_reference = float(
        np.median(neighborhood_mass[neighborhood_mass > 0.0])
    )
    if not np.isfinite(mass_reference) or mass_reference <= 0.0:
        mass_reference = 1.0

    scarcity = np.sqrt(
        mass_reference / np.maximum(neighborhood_mass, eps)
    )
    scarcity = np.clip(scarcity, 0.82, 1.25)
    scarcity_term = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )
    scarcity_term = np.clip(scarcity_term, 0.90, 1.12)

    feasible_rank = np.exp(
        -ranks / max(1.0, 1.10 * np.sqrt(float(n)))
    )
    alternative_mass = np.sum(
        feasible_rank * valid,
        axis=1,
    )
    alternative_reference = float(
        np.median(alternative_mass[alternative_mass > 0.0])
    )
    if (
        not np.isfinite(alternative_reference)
        or alternative_reference <= 0.0
    ):
        alternative_reference = 1.0

    feasibility = np.sqrt(
        alternative_reference
        / np.maximum(alternative_mass, eps)
    )
    feasibility = np.clip(feasibility, 0.86, 1.18)
    feasibility_term = np.sqrt(
        feasibility[:, None] * feasibility[None, :]
    )
    feasibility_term = np.clip(feasibility_term, 0.92, 1.10)

    profile_norm = np.sqrt(
        np.sum(proximity * proximity, axis=1)
    )
    overlap = np.divide(
        proximity @ proximity.T,
        profile_norm[:, None] * profile_norm[None, :],
        out=np.zeros((n, n), dtype=float),
        where=(
            profile_norm[:, None] * profile_norm[None, :]
        ) > 0.0,
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge_term = 1.0 + 0.08 * (
        1.0 - overlap
    ) * np.exp(
        -0.70 * np.maximum(relative_distance - 1.0, 0.0)
    )
    bridge_term = np.clip(bridge_term, 0.96, 1.08)

    cluster_penalty = np.where(
        mutual_nearest & (overlap > 0.72),
        1.0 - 0.05 * overlap,
        1.0,
    )

    scores = (
        distance_term
        * (
            0.50
            + 0.30 * bidirectional_rank
            + 0.20 * endpoint_quality
        )
        * regret_term
        * mutual_term
        * long_edge_term
        * scarcity_term
        * feasibility_term
        * bridge_term
        * cluster_penalty
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
