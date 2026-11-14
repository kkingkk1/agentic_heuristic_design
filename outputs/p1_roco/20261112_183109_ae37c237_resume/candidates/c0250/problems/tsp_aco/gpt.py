import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return global-diversity and long-term-potential TSP edge priorities."""
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

    positive = d[valid & (d > 0.0)]
    all_values = d[valid]
    eps = np.finfo(float).eps

    if positive.size:
        scale = float(np.median(positive))
    elif all_values.size:
        scale = float(np.max(all_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    finite_max = float(np.max(all_values))
    penalty = 6.0 * max(finite_max, scale) + scale
    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_work = np.take_along_axis(work, order, axis=1)

    best = sorted_work[:, 0]
    second = sorted_work[:, 1] if n > 2 else best
    best = np.where(np.isfinite(best), best, scale)
    second = np.where(np.isfinite(second), second, best)

    local_scale = 0.5 * (best + second)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        scale,
    )
    local_scale = np.maximum(local_scale, eps)

    relative = np.divide(
        work,
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(work),
    )

    distance_term = np.exp(-np.clip(1.05 * relative, 0.0, 60.0))

    rank_scale = max(1.0, 0.85 * np.sqrt(float(n)))
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_term = np.exp(
        -0.5
        * (
            ranks / rank_scale
            + ranks.T / rank_scale
        )
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
        0.18 + 0.82 * endpoint_quality,
        0.18,
        1.0,
    )

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
    )
    regret = np.clip(regret, 0.0, 1.0)
    regret_term = 1.0 + 0.24 * (regret[:, None] + regret[None, :])

    nearest = order[:, 0]
    idx = np.arange(n)
    mutual = (
        (nearest[:, None] == idx[None, :])
        & (nearest[None, :] == idx[:, None])
    )
    mutual_term = np.where(mutual, 1.16, 1.0)

    affinity = np.exp(-np.clip(work / local_scale[:, None], 0.0, 50.0))
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
        np.divide(mean_mass, np.maximum(node_mass, eps))
    )
    scarcity = np.clip(scarcity, 0.86, 1.20)
    scarcity_term = np.sqrt(scarcity[:, None] * scarcity[None, :])

    probability = np.divide(
        affinity,
        np.maximum(node_mass[:, None], eps),
    )

    degree = np.sum(valid, axis=1)
    log_degree = np.log(np.maximum(degree, 2.0))
    entropy = -np.sum(
        np.where(
            probability > 0.0,
            probability * np.log(np.maximum(probability, eps)),
            0.0,
        ),
        axis=1,
    )
    entropy = np.clip(
        np.divide(entropy, log_degree, out=np.zeros_like(entropy), where=log_degree > 0.0),
        0.0,
        1.0,
    )

    diversity_need = 1.0 + 0.14 * (1.0 - entropy)
    diversity_term = np.sqrt(
        diversity_need[:, None] * diversity_need[None, :]
    )

    neighborhood_similarity = probability @ probability.T
    neighborhood_similarity = np.clip(neighborhood_similarity, 0.0, 1.0)
    cross_region = 1.0 - neighborhood_similarity

    two_step = probability @ probability.T
    two_step_max = np.maximum(
        np.max(two_step, axis=1, keepdims=True),
        eps,
    )
    two_step_normalized = np.divide(
        two_step,
        np.sqrt(two_step_max * two_step_max.T),
    )
    two_step_normalized = np.clip(two_step_normalized, 0.0, 1.0)

    affinity_order = np.argsort(-affinity, axis=1, kind="stable")
    top_affinity = np.take_along_axis(
        affinity,
        affinity_order[:, : min(2, n)],
        axis=1,
    )

    first_alt = top_affinity[:, 0]
    second_alt = (
        top_affinity[:, 1]
        if top_affinity.shape[1] > 1
        else np.zeros(n, dtype=float)
    )
    first_neighbor = affinity_order[:, 0]

    alternative = np.where(
        first_neighbor[:, None] == np.arange(n)[None, :],
        second_alt[:, None],
        first_alt[:, None],
    )
    alternative = np.broadcast_to(alternative, (n, n)).copy()

    alternative[~valid] = 0.0
    alternative_max = float(np.max(alternative))
    if alternative_max > eps:
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

    long_term_potential = (
        0.52 * continuation
        + 0.28 * cross_region
        + 0.20 * two_step_normalized
    )
    long_term_potential = np.clip(long_term_potential, 0.0, 1.0)

    neighbor_count = min(
        n - 1,
        max(2, int(np.sqrt(float(n)))),
    )
    neighbor_mask = np.zeros((n, n), dtype=bool)
    neighbor_mask[
        np.arange(n)[:, None],
        order[:, :neighbor_count],
    ] = True
    neighbor_mask &= valid

    shared_neighbors = (
        neighbor_mask.astype(float)
        @ neighbor_mask.astype(float).T
    )
    overlap = np.divide(
        shared_neighbors,
        float(max(1, neighbor_count)),
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    cycle_risk = overlap * (1.0 - cross_region)
    cycle_penalty = np.exp(
        -0.30 * cycle_risk * quality_gate
    )

    global_reward = (
        1.0
        + quality_gate
        * (
            0.34 * (scarcity_term - 1.0)
            + 0.22 * (diversity_term - 1.0)
            + 0.54 * long_term_potential
            + 0.28 * cross_region
        )
    )

    distance_penalty = np.exp(
        -0.48 * np.maximum(relative - 1.0, 0.0)
    )

    scores = (
        distance_term
        * (
            0.46
            + 0.31 * rank_term
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
