import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse, long-term TSP edge priorities."""
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

    valid_values = d[valid]
    positive_values = valid_values[valid_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif valid_values.size:
        global_scale = float(np.max(valid_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(valid_values))
    invalid_penalty = 8.0 * max(finite_max, global_scale) + global_scale

    work = np.where(valid, d, invalid_penalty)
    np.fill_diagonal(work, invalid_penalty)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(work, order, axis=1)

    best = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 2 else best

    best = np.where(np.isfinite(best), best, global_scale)
    second = np.where(np.isfinite(second), second, best)

    local_scale = 0.5 * (best + second)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    relative = np.divide(
        work,
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(work),
    )

    distance_term = np.exp(-np.clip(1.05 * relative, 0.0, 60.0))

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.9 * np.sqrt(float(n)))
    rank_term = np.exp(
        -0.5
        * (ranks / rank_scale + ranks.T / rank_scale)
    )

    endpoint_quality = np.exp(
        -0.5
        * np.clip(
            work / local_scale[:, None]
            + work / local_scale[None, :],
            0.0,
            60.0,
        )
    )

    quality_gate = np.clip(
        0.16 + 0.84 * endpoint_quality,
        0.16,
        1.0,
    )

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
    )
    regret = np.clip(regret, 0.0, 1.0)
    regret_term = 1.0 + 0.24 * (
        regret[:, None] + regret[None, :]
    )

    affinity = np.exp(
        -np.clip(work / local_scale[:, None], 0.0, 50.0)
    )
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    node_mass = np.sum(affinity, axis=1)
    positive_mass = node_mass[node_mass > 0.0]
    mean_mass = (
        float(np.mean(positive_mass))
        if positive_mass.size
        else 1.0
    )

    scarcity = np.sqrt(
        np.divide(
            mean_mass,
            np.maximum(node_mass, eps),
        )
    )
    scarcity = np.clip(scarcity, 0.84, 1.24)
    scarcity_term = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )

    probability = np.divide(
        affinity,
        np.maximum(node_mass[:, None], eps),
    )

    degree = np.sum(valid, axis=1)
    log_degree = np.log(np.maximum(degree, 2.0))

    entropy = -np.sum(
        np.where(
            probability > 0.0,
            probability * np.log(
                np.maximum(probability, eps)
            ),
            0.0,
        ),
        axis=1,
    )
    entropy = np.clip(
        np.divide(
            entropy,
            log_degree,
            out=np.zeros_like(entropy),
            where=log_degree > 0.0,
        ),
        0.0,
        1.0,
    )

    diversity_need = 1.0 + 0.16 * (1.0 - entropy)
    diversity_term = np.sqrt(
        diversity_need[:, None] * diversity_need[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)

    mutual = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_term = np.where(mutual, 1.15, 1.0)

    neighborhood_similarity = probability @ probability.T
    neighborhood_similarity = np.clip(
        neighborhood_similarity,
        0.0,
        1.0,
    )
    cross_region = 1.0 - neighborhood_similarity

    two_step = probability @ probability
    two_step = np.maximum(two_step, 0.0)

    two_step_scale = np.maximum(
        np.max(two_step, axis=1, keepdims=True),
        eps,
    )
    two_step_normalized = np.divide(
        two_step,
        np.sqrt(two_step_scale * two_step_scale.T),
    )
    two_step_normalized = np.clip(
        two_step_normalized,
        0.0,
        1.0,
    )

    masked_affinity = affinity.copy()
    masked_affinity[~valid] = -np.inf
    np.fill_diagonal(masked_affinity, -np.inf)

    affinity_order = np.argsort(
        -masked_affinity,
        axis=1,
        kind="stable",
    )
    sorted_affinity = np.take_along_axis(
        masked_affinity,
        affinity_order,
        axis=1,
    )

    first_value = np.where(
        np.isfinite(sorted_affinity[:, 0]),
        sorted_affinity[:, 0],
        0.0,
    )
    second_value = (
        np.where(
            np.isfinite(sorted_affinity[:, 1]),
            sorted_affinity[:, 1],
            0.0,
        )
        if n > 2
        else np.zeros(n, dtype=float)
    )

    first_neighbor = affinity_order[:, 0]
    alternative = np.where(
        first_neighbor[:, None] == indices[None, :],
        second_value[:, None],
        first_value[:, None],
    )

    alternative[~valid] = 0.0

    alternative_max = float(np.max(alternative))
    if np.isfinite(alternative_max) and alternative_max > eps:
        alternative_normalized = (
            alternative / alternative_max
        )
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

    neighbor_count = min(
        n - 1,
        max(2, int(np.ceil(np.sqrt(float(n))))),
    )

    neighbor_mask = np.zeros((n, n), dtype=bool)
    neighbor_mask[
        np.arange(n)[:, None],
        order[:, :neighbor_count],
    ] = True
    neighbor_mask &= valid

    neighborhood = neighbor_mask.astype(float)
    shared_neighbors = neighborhood @ neighborhood.T
    overlap = np.divide(
        shared_neighbors,
        float(max(1, neighbor_count)),
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    cycle_risk = overlap * (1.0 - cross_region)
    cycle_penalty = np.exp(
        -0.34 * cycle_risk * quality_gate
    )

    long_term_potential = (
        0.56 * continuation
        + 0.26 * cross_region
        + 0.18 * two_step_normalized
    )
    long_term_potential = np.clip(
        long_term_potential,
        0.0,
        1.0,
    )

    global_reward = 1.0 + quality_gate * (
        0.30 * (scarcity_term - 1.0)
        + 0.24 * (diversity_term - 1.0)
        + 0.58 * long_term_potential
        + 0.24 * cross_region
    )

    distance_penalty = np.exp(
        -0.46 * np.maximum(relative - 1.0, 0.0)
    )

    scores = (
        distance_term
        * (
            0.45
            + 0.32 * rank_term
            + 0.23 * endpoint_quality
        )
        * regret_term
        * mutual_term
        * distance_penalty
        * cycle_penalty
        * global_reward
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
