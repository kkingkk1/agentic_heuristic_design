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
    finite = np.isfinite(distances)
    nonnegative = distances >= 0.0
    raw_valid = finite & nonnegative & ~diagonal

    if not np.any(raw_valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(raw_valid, distances, np.inf)
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

    row_order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    valid_rank = np.where(valid, row_rank, np.inf)

    available_degree = valid.sum(axis=1).astype(float)
    density = float(valid.sum()) / max(float(n * (n - 1)), 1.0)
    sparsity = 1.0 - density

    usable = min(7, n - 1)
    neighborhood = np.take_along_axis(
        safe,
        row_order[:, :usable],
        axis=1,
    )
    neighborhood_valid = np.isfinite(neighborhood)

    local_scale = np.sum(
        np.where(neighborhood_valid, neighborhood, 0.0),
        axis=1,
    ) / np.maximum(neighborhood_valid.sum(axis=1), 1)

    local_scale = np.where(
        neighborhood_valid.any(axis=1),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    scale_ratio = local_scale / global_scale
    log_scale = np.log(np.maximum(scale_ratio, eps))
    heterogeneity = float(np.std(log_scale))
    heterogeneity_signal = np.clip(heterogeneity / 1.5, 0.0, 1.0)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = safe / np.maximum(pair_scale, eps)

    local_quality = np.exp(
        -np.clip(normalized_distance, 0.0, 50.0)
    )
    global_quality = np.exp(
        -np.clip(safe / global_scale, 0.0, 50.0)
    )

    rank_temperature = max(float(min(5, n - 1)), 1.0)
    rank_quality = np.exp(
        -np.clip(valid_rank / rank_temperature, 0.0, 50.0)
    )
    reciprocal_rank = np.sqrt(rank_quality * rank_quality.T)

    endpoint_affinity = np.sqrt(
        np.exp(-np.clip(safe / local_scale[:, None], 0.0, 50.0))
        * np.exp(-np.clip(safe / local_scale[None, :], 0.0, 50.0))
    )

    scales = (3, 5, 7)
    masks = []
    overlaps = []

    for k in scales:
        kk = min(k, n - 1)
        mask = np.zeros((n, n), dtype=bool)
        mask[
            np.arange(n)[:, None],
            row_order[:, :kk],
        ] = True
        mask &= valid
        masks.append(mask)

        degree = np.maximum(mask.sum(axis=1).astype(float), 1.0)
        overlap = mask.astype(float) @ mask.astype(float).T
        overlap /= np.sqrt(degree[:, None] * degree[None, :])
        overlaps.append(np.clip(overlap, 0.0, 1.0))

    candidate_support = np.mean(
        np.stack([mask.astype(float) for mask in masks], axis=0),
        axis=0,
    )
    neighborhood_overlap = np.mean(
        np.stack(overlaps, axis=0),
        axis=0,
    )

    mutual_support = np.sqrt(
        candidate_support * candidate_support.T
    )

    # Opportunity penalty: highly overlapping neighborhoods indicate
    # redundant edges that concentrate sampling in one local cluster.
    opportunity_penalty = 1.0 - 0.20 * np.clip(
        neighborhood_overlap * mutual_support,
        0.0,
        1.0,
    )

    # Bridge potential rewards edges joining locally distinct regions.
    bridge_potential = 1.0 + 0.10 * (
        1.0 - neighborhood_overlap
    ) * (0.35 + 0.65 * mutual_support)

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
    regret = np.clip(regret, 0.0, 5.0)

    endpoint_regret = np.sqrt(
        (1.0 + regret[:, None]) * (1.0 + regret[None, :])
    )
    regret_weight = 0.07 + 0.10 * (
        0.5 * sparsity + 0.5 * heterogeneity_signal
    )
    long_term_factor = 1.0 + regret_weight * np.clip(
        endpoint_regret - 1.0,
        0.0,
        3.0,
    ) * (0.35 + 0.65 * mutual_support)

    local_weight = 0.58 - 0.10 * heterogeneity_signal
    global_weight = 0.22 + 0.10 * heterogeneity_signal
    affinity_weight = 1.0 - local_weight - global_weight

    quality = (
        local_weight * local_quality
        + global_weight * global_quality
        + affinity_weight * endpoint_affinity
    )

    rank_factor = 0.58 + 0.42 * reciprocal_rank
    mutual_factor = 0.92 + 0.08 * mutual_support

    scores = (
        quality
        * rank_factor
        * mutual_factor
        * opportunity_penalty
        * bridge_potential
        * long_term_factor
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)

    # Globally balance row mass. The correction is deliberately bounded and
    # symmetric so neglected vertices receive support without flattening all
    # distance information.
    row_mass = scores.sum(axis=1)
    positive_mass = row_mass[row_mass > 0.0]

    if positive_mass.size:
        target_mass = float(np.median(positive_mass))
        balance_strength = 0.18 + 0.14 * (
            sparsity + heterogeneity_signal
        )

        for _ in range(4):
            row_mass = scores.sum(axis=1)
            safe_mass = np.maximum(row_mass, eps)
            ratio = np.sqrt(target_mass / safe_mass)
            ratio = np.clip(
                ratio,
                1.0 - balance_strength,
                1.0 + balance_strength,
            )
            scores *= np.sqrt(ratio[:, None] * ratio[None, :])
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
