import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    raw_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(raw_valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(raw_valid, distances, np.inf)

    # Conservative symmetrization: an edge is usable only when at least
    # one directed observation is finite and nonnegative.
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    valid_values = safe[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(valid_values)), 1.0)

    global_scale = max(global_scale, eps)

    # Stable ranks among finite neighbors.
    row_order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)
    valid_rank = np.where(valid, row_rank, np.inf)

    degree = valid.sum(axis=1).astype(float)
    density = float(valid.sum()) / max(float(n * (n - 1)), 1.0)
    sparsity = 1.0 - density

    # Node-adaptive scale, robust to very different distance magnitudes.
    neighborhood_size = min(max(4, int(np.sqrt(max(n, 2))) + 2), n - 1)
    nearest_values = np.take_along_axis(
        safe,
        row_order[:, :neighborhood_size],
        axis=1,
    )
    nearest_valid = np.isfinite(nearest_values)

    local_scale = np.sum(
        np.where(nearest_valid, nearest_values, 0.0),
        axis=1,
    ) / np.maximum(nearest_valid.sum(axis=1), 1.0)

    local_scale = np.where(
        nearest_valid.any(axis=1),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    log_scale = np.log(np.maximum(local_scale / global_scale, eps))
    heterogeneity = float(np.std(log_scale))
    heterogeneity_signal = float(np.clip(heterogeneity / 1.5, 0.0, 1.0))

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = safe / np.maximum(pair_scale, eps)
    global_distance = safe / global_scale

    local_quality = np.exp(-np.clip(normalized_distance, 0.0, 50.0))
    global_quality = np.exp(-np.clip(global_distance, 0.0, 50.0))

    rank_temperature = max(float(min(5, n - 1)), 1.0)
    rank_quality = np.exp(
        -np.clip(valid_rank / rank_temperature, 0.0, 50.0)
    )
    reciprocal_rank = np.sqrt(rank_quality * rank_quality.T)

    # Endpoint affinity is deliberately bounded and kept separate for ablation.
    endpoint_affinity = np.sqrt(
        np.exp(-np.clip(safe / local_scale[:, None], 0.0, 50.0))
        * np.exp(-np.clip(safe / local_scale[None, :], 0.0, 50.0))
    )

    # Multi-scale candidate neighborhoods.
    scales = tuple(
        sorted(
            set(
                min(k, n - 1)
                for k in (3, 5, 8, max(3, int(np.sqrt(max(n, 2))) + 1))
            )
        )
    )

    masks = []
    overlaps = []

    for k in scales:
        mask = np.zeros((n, n), dtype=bool)
        mask[
            np.arange(n)[:, None],
            row_order[:, :k],
        ] = True
        mask &= valid
        masks.append(mask)

        local_degree = np.maximum(mask.sum(axis=1).astype(float), 1.0)
        common = mask.astype(float) @ mask.astype(float).T
        normalized_common = common / np.sqrt(
            local_degree[:, None] * local_degree[None, :]
        )
        overlaps.append(np.clip(normalized_common, 0.0, 1.0))

    candidate_support = np.mean(
        np.stack([m.astype(float) for m in masks], axis=0),
        axis=0,
    )
    neighborhood_overlap = np.mean(
        np.stack(overlaps, axis=0),
        axis=0,
    )
    mutual_support = np.sqrt(
        candidate_support * candidate_support.T
    )

    # Low overlap indicates a possible bridge between local regions.
    bridge_signal = (
        (1.0 - neighborhood_overlap)
        * (0.35 + 0.65 * mutual_support)
    )

    # Opportunity cost: redundant edges receive a mild, bounded penalty.
    opportunity_cost = np.clip(
        neighborhood_overlap * (0.35 + 0.65 * mutual_support),
        0.0,
        1.0,
    )
    opportunity_factor = 1.0 - (
        0.16 + 0.10 * sparsity
    ) * opportunity_cost

    # Long-term potential from endpoint regret and edge scarcity.
    nearest = np.full(n, np.inf, dtype=float)
    second_nearest = np.full(n, np.inf, dtype=float)

    for i in range(n):
        row_values = safe[i, valid[i]]
        if row_values.size:
            ordered = np.sort(row_values)
            nearest[i] = ordered[0]
            second_nearest[i] = ordered[min(1, ordered.size - 1)]

    regret = np.maximum(second_nearest - nearest, 0.0)
    regret /= np.maximum(local_scale, eps)
    regret = np.clip(regret, 0.0, 6.0)

    degree_reference = max(float(np.median(degree[degree > 0])), 1.0)
    scarcity = np.clip(
        degree_reference / np.maximum(degree, 1.0),
        0.5,
        2.0,
    )

    endpoint_regret = np.sqrt(
        (1.0 + regret[:, None]) * (1.0 + regret[None, :])
    )
    endpoint_scarcity = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )

    long_term_signal = np.clip(
        0.55 * (endpoint_regret - 1.0)
        + 0.45 * (endpoint_scarcity - 1.0),
        -0.5,
        3.0,
    )

    long_term_weight = 0.07 + 0.10 * (
        0.55 * sparsity + 0.45 * heterogeneity_signal
    )
    long_term_factor = np.exp(
        np.clip(long_term_weight * long_term_signal, -0.20, 0.35)
    )

    # Log-space fusion reduces over-concentration from repeated multiplication.
    local_weight = 0.52 - 0.10 * heterogeneity_signal
    global_weight = 0.20 + 0.10 * heterogeneity_signal
    affinity_weight = 1.0 - local_weight - global_weight

    quality = (
        local_weight * local_quality
        + global_weight * global_quality
        + affinity_weight * endpoint_affinity
    )
    quality = np.clip(quality, eps, None)

    rank_factor = 0.62 + 0.38 * reciprocal_rank
    support_factor = 0.94 + 0.06 * mutual_support
    diversity_factor = (
        1.0
        + (0.08 + 0.07 * sparsity) * bridge_signal
    )

    log_scores = (
        np.log(quality)
        + np.log(np.clip(rank_factor, eps, None))
        + np.log(np.clip(support_factor, eps, None))
        + np.log(np.clip(opportunity_factor, eps, None))
        + np.log(np.clip(diversity_factor, eps, None))
        + np.log(np.clip(long_term_factor, eps, None))
    )

    scores = np.exp(np.clip(log_scores, -50.0, 20.0))
    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)

    # Symmetric bounded balancing: redistribute prior mass away from hubs
    # while retaining local edge preferences.
    row_mass = scores.sum(axis=1)
    positive_mass = row_mass[row_mass > 0.0]

    if positive_mass.size:
        target_mass = float(np.median(positive_mass))
        balance_strength = 0.16 + 0.16 * (
            0.55 * sparsity + 0.45 * heterogeneity_signal
        )

        for _ in range(5):
            row_mass = scores.sum(axis=1)
            safe_mass = np.maximum(row_mass, eps)
            correction = np.sqrt(target_mass / safe_mass)
            correction = np.clip(
                correction,
                1.0 - balance_strength,
                1.0 + balance_strength,
            )

            scores *= np.sqrt(
                correction[:, None] * correction[None, :]
            )
            scores[~valid] = 0.0
            scores = 0.5 * (scores + scores.T)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
