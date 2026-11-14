import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized, globally aware TSP edge priorities."""

    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite_nonnegative = (
        np.isfinite(distances) & (distances >= 0.0)
    )
    np.fill_diagonal(finite_nonnegative, False)

    valid = finite_nonnegative & finite_nonnegative.T
    if not np.any(valid):
        return result

    with np.errstate(all="ignore"):
        symmetric_distance = 0.5 * (
            distances + distances.T
        )

    symmetric_distance[~valid] = np.inf
    np.fill_diagonal(symmetric_distance, np.inf)

    valid_values = symmetric_distance[valid]
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

    positive_max = (
        float(np.max(positive_values))
        if positive_values.size
        else global_scale
    )
    invalid_value = (
        8.0 * max(positive_max, global_scale)
        + global_scale
    )

    work_distance = np.where(
        valid,
        symmetric_distance,
        invalid_value,
    )
    np.fill_diagonal(work_distance, invalid_value)

    order = np.argsort(
        work_distance,
        axis=1,
        kind="stable",
    )
    ordered_distance = np.take_along_axis(
        work_distance,
        order,
        axis=1,
    )

    nearest = ordered_distance[:, 0]
    second_nearest = (
        ordered_distance[:, 1]
        if n > 2
        else nearest
    )

    nearest = np.where(
        np.isfinite(nearest),
        nearest,
        global_scale,
    )
    second_nearest = np.where(
        np.isfinite(second_nearest),
        second_nearest,
        nearest,
    )

    local_scale = 0.5 * (
        np.maximum(nearest, 0.0)
        + np.maximum(second_nearest, 0.0)
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    relative_distance = np.divide(
        work_distance,
        np.sqrt(
            local_scale[:, None] * local_scale[None, :]
        ),
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(work_distance),
    )

    distance_quality = np.exp(
        -np.clip(1.08 * relative_distance, 0.0, 60.0)
    )

    endpoint_quality = np.exp(
        -0.5
        * np.clip(
            work_distance / local_scale[:, None]
            + work_distance / local_scale[None, :],
            0.0,
            60.0,
        )
    )

    ranks = np.empty((n, n), dtype=float)
    ranks[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    rank_scale = max(
        1.0,
        0.85 * np.sqrt(float(n)),
    )
    rank_quality = np.exp(
        -0.5
        * (
            ranks / rank_scale
            + ranks.T / rank_scale
        )
    )

    affinity = np.exp(
        -np.clip(
            work_distance / local_scale[:, None],
            0.0,
            50.0,
        )
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
    scarcity = np.clip(scarcity, 0.82, 1.28)
    scarcity_factor = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )

    transition = np.divide(
        affinity,
        np.maximum(node_mass[:, None], eps),
    )

    degree = np.sum(valid, axis=1).astype(float)
    entropy = -np.sum(
        np.where(
            transition > 0.0,
            transition * np.log(
                np.maximum(transition, eps)
            ),
            0.0,
        ),
        axis=1,
    )
    entropy_scale = np.log(
        np.maximum(degree, 2.0)
    )
    normalized_entropy = np.divide(
        entropy,
        entropy_scale,
        out=np.zeros_like(entropy),
        where=entropy_scale > 0.0,
    )
    normalized_entropy = np.clip(
        normalized_entropy,
        0.0,
        1.0,
    )

    diversity_need = 1.0 + 0.18 * (
        1.0 - normalized_entropy
    )
    diversity_factor = np.sqrt(
        diversity_need[:, None]
        * diversity_need[None, :]
    )

    nearest_neighbor = order[:, 0]
    indices = np.arange(n)

    mutual_nearest = (
        nearest_neighbor[:, None] == indices[None, :]
    ) & (
        nearest_neighbor[None, :] == indices[:, None]
    )
    mutual_factor = np.where(
        mutual_nearest,
        1.17,
        1.0,
    )

    neighborhood_similarity = transition @ transition.T
    neighborhood_similarity = np.clip(
        neighborhood_similarity,
        0.0,
        1.0,
    )
    cross_region = 1.0 - neighborhood_similarity

    two_step_reachability = transition @ transition
    two_step_reachability = np.maximum(
        two_step_reachability,
        0.0,
    )

    row_max = np.maximum(
        np.max(two_step_reachability, axis=1),
        eps,
    )
    two_step_quality = np.divide(
        two_step_reachability,
        np.sqrt(row_max[:, None] * row_max[None, :]),
    )
    two_step_quality = np.clip(
        two_step_quality,
        0.0,
        1.0,
    )

    affinity_for_sorting = affinity.copy()
    affinity_for_sorting[~valid] = -np.inf
    np.fill_diagonal(affinity_for_sorting, -np.inf)

    affinity_order = np.argsort(
        -affinity_for_sorting,
        axis=1,
        kind="stable",
    )
    ordered_affinity = np.take_along_axis(
        affinity_for_sorting,
        affinity_order,
        axis=1,
    )

    best_affinity = np.where(
        np.isfinite(ordered_affinity[:, 0]),
        ordered_affinity[:, 0],
        0.0,
    )

    if n > 2:
        second_affinity = np.where(
            np.isfinite(ordered_affinity[:, 1]),
            ordered_affinity[:, 1],
            0.0,
        )
    else:
        second_affinity = np.zeros(n, dtype=float)

    best_neighbor = affinity_order[:, 0]

    alternative_affinity = np.where(
        best_neighbor[:, None] == indices[None, :],
        second_affinity[:, None],
        best_affinity[:, None],
    )
    alternative_affinity[~valid] = 0.0

    alternative_max = float(
        np.max(alternative_affinity)
    )
    if np.isfinite(alternative_max) and alternative_max > eps:
        alternative_quality = (
            alternative_affinity / alternative_max
        )
    else:
        alternative_quality = np.zeros_like(
            alternative_affinity
        )

    continuation = np.sqrt(
        np.clip(
            alternative_quality
            * alternative_quality.T,
            0.0,
            1.0,
        )
    )

    regret = np.divide(
        second_nearest - nearest,
        np.maximum(second_nearest, eps),
    )
    regret = np.clip(regret, 0.0, 1.0)
    regret_factor = 1.0 + 0.26 * (
        regret[:, None] + regret[None, :]
    )

    neighbor_count = min(
        n - 1,
        max(2, int(np.ceil(np.sqrt(float(n))))),
    )

    neighborhood = np.zeros(
        (n, n),
        dtype=float,
    )
    neighborhood[
        np.arange(n)[:, None],
        order[:, :neighbor_count],
    ] = 1.0
    neighborhood *= valid

    shared_neighbors = neighborhood @ neighborhood.T
    overlap = np.divide(
        shared_neighbors,
        float(max(1, neighbor_count)),
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    cycle_risk = overlap * (
        1.0 - cross_region
    )
    cycle_penalty = np.exp(
        -0.38
        * cycle_risk
        * np.clip(endpoint_quality, 0.0, 1.0)
    )

    long_term_potential = np.clip(
        0.54 * continuation
        + 0.28 * cross_region
        + 0.18 * two_step_quality,
        0.0,
        1.0,
    )

    quality_gate = np.clip(
        0.14 + 0.86 * endpoint_quality,
        0.14,
        1.0,
    )

    global_reward = 1.0 + quality_gate * (
        0.28 * (scarcity_factor - 1.0)
        + 0.25 * (diversity_factor - 1.0)
        + 0.58 * long_term_potential
        + 0.22 * cross_region
    )

    long_edge_penalty = np.exp(
        -0.48
        * np.maximum(relative_distance - 1.0, 0.0)
    )

    scores = (
        distance_quality
        * (
            0.43
            + 0.34 * rank_quality
            + 0.23 * endpoint_quality
        )
        * regret_factor
        * mutual_factor
        * long_edge_penalty
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
