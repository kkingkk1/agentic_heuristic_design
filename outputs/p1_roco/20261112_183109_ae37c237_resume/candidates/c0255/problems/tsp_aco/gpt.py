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

    with np.errstate(all="ignore"):
        distances = 0.5 * (distances + distances.T)

    distances[~valid] = np.inf
    np.fill_diagonal(distances, np.inf)

    valid_values = distances[valid]
    positive_values = valid_values[valid_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.max(valid_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(valid_values))
    invalid_penalty = 5.0 * max(finite_max, global_scale) + global_scale

    work = np.where(valid, distances, invalid_penalty)
    np.fill_diagonal(work, invalid_penalty)

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

    denominator = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        work,
        denominator,
        out=np.full((n, n), np.inf, dtype=float),
        where=denominator > 0.0,
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

    quality_gate = 0.55 + 0.45 * endpoint_quality

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
    regret_bonus = 1.0 + 0.30 * (
        regret[:, None] + regret[None, :]
    )

    indices = np.arange(n)
    nearest = order[:, 0]

    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
        & valid
    )
    mutual_bonus = np.where(mutual_nearest, 1.22, 1.0)

    k = min(n - 1, max(2, int(np.ceil(np.sqrt(float(n))))))
    neighbor_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        row_candidates = order[i, valid[i, order[i]]]
        neighbor_mask[i, row_candidates[:k]] = True

    neighbor_mask &= valid

    local_affinity = np.exp(
        -np.clip(
            work / np.maximum(local_scale[:, None], eps),
            0.0,
            50.0,
        )
    )
    local_affinity[~valid] = 0.0
    np.fill_diagonal(local_affinity, 0.0)

    candidate_affinity = (
        local_affinity * neighbor_mask.astype(float)
    )

    candidate_mass = np.sum(candidate_affinity, axis=1)
    probability = np.divide(
        candidate_affinity,
        np.maximum(candidate_mass[:, None], eps),
    )

    positive_mass = candidate_mass[candidate_mass > 0.0]
    mean_mass = (
        float(np.mean(positive_mass))
        if positive_mass.size
        else 1.0
    )

    scarcity = np.sqrt(
        np.divide(
            mean_mass,
            np.maximum(candidate_mass, eps),
        )
    )
    scarcity = np.clip(scarcity, 0.84, 1.24)

    scarcity_bonus = 1.0 + 0.16 * (
        scarcity[:, None] + scarcity[None, :]
    ) * quality_gate

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

    degree = np.sum(valid, axis=1)
    entropy_scale = np.log(np.maximum(degree, 2.0))
    normalized_entropy = np.divide(
        entropy,
        entropy_scale,
        out=np.zeros_like(entropy),
        where=entropy_scale > 0.0,
    )
    normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)

    diversity_need = 1.0 + 0.14 * (
        1.0 - normalized_entropy
    )
    diversity_bonus = np.sqrt(
        diversity_need[:, None] * diversity_need[None, :]
    )

    neighborhood_difference = np.zeros((n, n), dtype=float)
    neighborhood_sizes = neighbor_mask.sum(axis=1).astype(float)
    overlap = (
        neighbor_mask.astype(float)
        @ neighbor_mask.astype(float).T
    )
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

    neighborhood_similarity = np.clip(
        probability @ probability.T,
        0.0,
        1.0,
    )
    cross_region = 1.0 - neighborhood_similarity

    two_step = np.maximum(probability @ probability, 0.0)
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

    if np.isfinite(continuation_max) and continuation_max > eps:
        continuation /= continuation_max

    alternative_affinity = candidate_affinity.copy()

    for i in range(n):
        nearest_i = nearest[i]
        if 0 <= nearest_i < n:
            alternative_affinity[i, nearest_i] = 0.0

    alternative_support = np.max(
        alternative_affinity,
        axis=1,
    )
    alternative_support = np.clip(
        alternative_support,
        0.0,
        1.0,
    )

    alternative_bonus = np.sqrt(
        alternative_support[:, None]
        * alternative_support[None, :]
    )

    long_term_potential = (
        0.55 * continuation
        + 0.25 * alternative_bonus
        + 0.20 * two_step_normalized
    )
    long_term_potential = np.clip(
        long_term_potential,
        0.0,
        1.0,
    )

    moderate_length_gate = np.exp(
        -0.90 * np.maximum(relative_distance - 1.10, 0.0)
    )

    bridge_gate = (
        moderate_length_gate
        * neighborhood_difference
        * cross_region
        * quality_gate
        * alternative_bonus
    )

    bridge_bonus = 1.0 + 0.09 * bridge_gate

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
        * (1.0 - 0.5 * bidirectional_rank)
        * (1.0 - 0.5 * cross_region)
        * quality_gate
    )
    closure_pressure = np.clip(
        closure_pressure,
        0.0,
        1.0,
    )

    cycle_penalty = 1.0 - 0.07 * closure_pressure

    global_bonus = 1.0 + quality_gate * (
        0.16 * (diversity_bonus - 1.0)
        + 0.20 * long_term_potential
        + 0.08 * cross_region
    )

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
        * bridge_bonus
        * global_bonus
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
