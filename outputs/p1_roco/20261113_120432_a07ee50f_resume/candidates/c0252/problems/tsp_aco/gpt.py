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
    valid_directed = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    safe = np.where(valid_directed, distances, np.inf)
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

    masked = np.where(valid, safe, np.inf)
    row_order = np.argsort(masked, axis=1, kind="stable")
    column_order = np.argsort(masked, axis=0, kind="stable")

    usable_k = min(3, n - 1)
    nearest = np.take_along_axis(
        masked,
        row_order[:, :usable_k],
        axis=1,
    )
    nearest_valid = np.isfinite(nearest)

    sorted_nearest = np.sort(
        np.where(nearest_valid, nearest, np.inf),
        axis=1,
    )
    scale_count = min(2, usable_k)
    scale_values = sorted_nearest[:, :scale_count]
    scale_valid = np.isfinite(scale_values)

    local_scale = np.sum(
        np.where(scale_valid, scale_values, 0.0),
        axis=1,
    ) / np.maximum(scale_valid.sum(axis=1), 1)

    local_scale = np.where(
        scale_valid.any(axis=1),
        local_scale,
        global_scale,
    )

    positive_local = local_scale[local_scale > eps]
    fallback_scale = (
        float(np.median(positive_local))
        if positive_local.size
        else global_scale
    )
    local_scale = np.maximum(local_scale, fallback_scale * 1e-6)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = np.divide(
        safe,
        pair_scale,
        out=np.full_like(safe, np.inf, dtype=float),
        where=valid,
    )

    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    column_rank = np.empty((n, n), dtype=float)
    column_rank[
        column_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    distance_signal = -np.log1p(
        np.clip(normalized_distance, 0.0, 1e6)
    )

    rank_signal = -0.5 * (
        np.log1p(row_rank) + np.log1p(column_rank)
    )
    rank_signal = np.maximum(rank_signal, -np.log1p(float(n)))

    endpoint_ratio_i = np.divide(
        safe,
        local_scale[:, None],
        out=np.full_like(safe, np.inf, dtype=float),
        where=valid,
    )
    endpoint_ratio_j = np.divide(
        safe,
        local_scale[None, :],
        out=np.full_like(safe, np.inf, dtype=float),
        where=valid,
    )
    endpoint_affinity = -0.5 * (
        np.log1p(np.clip(endpoint_ratio_i, 0.0, 1e6))
        + np.log1p(np.clip(endpoint_ratio_j, 0.0, 1e6))
    )

    alternative_scale = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    relative_regret = np.divide(
        safe,
        np.maximum(alternative_scale, eps),
        out=np.full_like(safe, np.inf, dtype=float),
        where=valid,
    ) - 1.0
    regret_penalty = -0.12 * np.clip(relative_regret, 0.0, 4.0)

    degree = valid.sum(axis=1).astype(float)
    top_k = min(2, n - 1)

    top_neighbors = np.zeros((n, n), dtype=bool)
    top_neighbors[
        np.arange(n)[:, None],
        row_order[:, :top_k],
    ] = True

    mutual_top = (
        valid
        & top_neighbors
        & top_neighbors.T
    ).astype(float)

    nearest_row = row_order[:, 0]
    nearest_column = column_order[0, :]
    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_column[None, :] == np.arange(n)[:, None])
    ).astype(float)

    mutual_signal = 0.10 * mutual_top + 0.14 * mutual_nearest

    sparse_endpoint = (
        (degree[:, None] <= top_k)
        | (degree[None, :] <= top_k)
    ).astype(float)
    sparse_protection = (
        0.10
        * sparse_endpoint
        * np.exp(-np.clip(normalized_distance, 0.0, 12.0))
    )

    scales = sorted(
        set(
            min(k, n - 1)
            for k in (2, 3, max(3, int(np.sqrt(max(n, 2))) + 1))
        )
    )

    neighborhood_masks = []
    for k in scales:
        mask = np.zeros((n, n), dtype=bool)
        mask[
            np.arange(n)[:, None],
            row_order[:, :k],
        ] = True
        neighborhood_masks.append(mask & valid)

    support = np.mean(
        np.stack(
            [mask.astype(float) for mask in neighborhood_masks],
            axis=0,
        ),
        axis=0,
    )
    mutual_support = np.sqrt(support * support.T)

    overlaps = []
    for mask in neighborhood_masks:
        local_degree = np.maximum(mask.sum(axis=1).astype(float), 1.0)
        common = mask.astype(float) @ mask.astype(float).T
        overlap = common / np.sqrt(
            local_degree[:, None] * local_degree[None, :]
        )
        overlaps.append(np.clip(overlap, 0.0, 1.0))

    neighborhood_overlap = np.mean(
        np.stack(overlaps, axis=0),
        axis=0,
    )

    bridge_signal = (
        (1.0 - neighborhood_overlap)
        * (0.35 + 0.65 * mutual_support)
    )
    bridge_reward = (
        0.035
        * (1.0 + 0.25 * (1.0 - degree[:, None] / np.maximum(degree.max(), 1.0)))
        * bridge_signal
    )

    nearest_value = np.min(masked, axis=1)
    second_value = np.partition(masked, 1, axis=1)[:, 1]
    nearest_value = np.where(np.isfinite(nearest_value), nearest_value, local_scale)
    second_value = np.where(np.isfinite(second_value), second_value, nearest_value)

    endpoint_regret = np.clip(
        (second_value - nearest_value) / np.maximum(local_scale, eps),
        0.0,
        6.0,
    )
    scarcity = np.clip(
        np.median(degree[degree > 0]) / np.maximum(degree, 1.0)
        if np.any(degree > 0)
        else np.ones(n),
        0.5,
        2.0,
    )

    long_term_signal = (
        0.55 * (
            np.sqrt(
                (1.0 + endpoint_regret[:, None])
                * (1.0 + endpoint_regret[None, :])
            ) - 1.0
        )
        + 0.45 * (
            np.sqrt(scarcity[:, None] * scarcity[None, :]) - 1.0
        )
    )
    long_term_reward = 0.035 * np.clip(long_term_signal, -0.5, 2.0)

    log_scores = (
        0.48 * distance_signal
        + 0.21 * rank_signal
        + 0.21 * endpoint_affinity
        + regret_penalty
        + mutual_signal
        + sparse_protection
        + bridge_reward
        + long_term_reward
    )
    log_scores[~valid] = -np.inf

    finite_scores = np.where(np.isfinite(log_scores), log_scores, 0.0)
    valid_max = float(np.max(finite_scores[valid]))

    scores = np.zeros((n, n), dtype=float)
    scores[valid] = np.exp(
        np.clip(finite_scores[valid] - valid_max, -50.0, 0.0)
    )

    scores = 0.5 * (scores + scores.T)

    positive_mass = scores.sum(axis=1)
    positive = positive_mass > 0.0
    if np.any(positive):
        target_mass = float(np.median(positive_mass[positive]))
        for _ in range(3):
            row_mass = np.maximum(scores.sum(axis=1), eps)
            correction = np.sqrt(target_mass / row_mass)
            correction = np.clip(correction, 0.92, 1.08)
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
