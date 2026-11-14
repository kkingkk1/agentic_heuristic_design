import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(directed_valid):
        return result

    raw = np.where(directed_valid, distances, np.inf)
    safe = np.minimum(raw, raw.T)
    np.fill_diagonal(safe, np.inf)

    valid = np.isfinite(safe)
    if not np.any(valid):
        return result

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    global_scale = max(global_scale, eps)

    valid_counts = valid.sum(axis=1).astype(float)

    order = np.argsort(safe, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(safe, order, axis=1)
    sorted_valid = np.isfinite(sorted_distances)

    usable = min(6, n - 1)
    local_distances = sorted_distances[:, :usable]
    local_valid = sorted_valid[:, :usable]

    local_scale = np.sum(
        np.where(local_valid, local_distances, 0.0),
        axis=1,
    ) / np.maximum(local_valid.sum(axis=1), 1)

    local_scale = np.where(
        local_valid.any(axis=1),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(
        local_scale,
        0.05 * global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = np.divide(
        safe,
        np.maximum(pair_scale, eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    normalized_distance = np.clip(normalized_distance, 0.0, 50.0)

    endpoint_ratio_i = np.divide(
        safe,
        np.maximum(local_scale[:, None], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    endpoint_ratio_j = np.divide(
        safe,
        np.maximum(local_scale[None, :], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )

    endpoint_ratio_i = np.clip(endpoint_ratio_i, 0.0, 50.0)
    endpoint_ratio_j = np.clip(endpoint_ratio_j, 0.0, 50.0)

    distance_score = np.exp(-1.15 * normalized_distance)
    endpoint_affinity = np.sqrt(
        np.exp(-endpoint_ratio_i)
        * np.exp(-endpoint_ratio_j)
    )

    rank = np.full((n, n), float(n), dtype=float)
    rank_values = np.arange(n, dtype=float)
    rank[np.arange(n)[:, None], order] = rank_values[None, :]

    rank_score = 1.0 / np.sqrt(
        (1.0 + rank) * (1.0 + rank.T)
    )

    short_k = min(3, n - 1)
    long_k = min(6, n - 1)

    short_neighbors = order[:, :short_k]
    long_neighbors = order[:, :long_k]

    short_mask = np.zeros((n, n), dtype=bool)
    long_mask = np.zeros((n, n), dtype=bool)

    np.put_along_axis(
        short_mask,
        short_neighbors,
        True,
        axis=1,
    )
    np.put_along_axis(
        long_mask,
        long_neighbors,
        True,
        axis=1,
    )

    short_mask &= valid
    long_mask &= valid

    mutual_short = (
        short_mask
        & short_mask.T
        & valid
    ).astype(float)

    mutual_long = (
        long_mask
        & long_mask.T
        & valid
    ).astype(float)

    nearest = order[:, 0]
    nearest_valid = np.isfinite(
        np.take_along_axis(safe, order[:, :1], axis=1)
    )[:, 0]

    mutual_nearest = (
        valid
        & nearest_valid[:, None]
        & nearest_valid[None, :]
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    logits = (
        -1.05 * normalized_distance
        -0.10 * endpoint_ratio_i
    )

    logits = np.where(valid, logits, -np.inf)
    row_max = np.max(
        np.where(valid, logits, -np.inf),
        axis=1,
        keepdims=True,
    )

    stable_logits = np.where(
        valid,
        logits - row_max,
        -np.inf,
    )
    weights = np.where(
        valid,
        np.exp(np.clip(stable_logits, -60.0, 0.0)),
        0.0,
    )

    weight_sum = weights.sum(axis=1, keepdims=True)
    preferences = np.divide(
        weights,
        np.maximum(weight_sum, eps),
        out=np.zeros_like(weights),
        where=weight_sum > 0.0,
    )

    mutual_preference = np.sqrt(
        preferences * preferences.T
    )

    second_distance = np.full(n, global_scale, dtype=float)
    if long_k >= 2:
        second_values = sorted_distances[:, 1]
        second_valid = np.isfinite(second_values)
        second_distance = np.where(
            second_valid,
            second_values,
            global_scale,
        )

    scarcity_ratio = np.divide(
        np.sqrt(
            np.maximum(second_distance[:, None], global_scale)
            * np.maximum(second_distance[None, :], global_scale)
        ),
        safe + 0.5 * global_scale,
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )

    scarcity_bonus = 1.0 + 0.16 * np.clip(
        scarcity_ratio - 0.7,
        0.0,
        2.5,
    )

    opportunity_ratio = np.divide(
        safe,
        np.maximum(
            np.minimum(
                local_scale[:, None],
                local_scale[None, :],
            ),
            eps,
        ),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )

    opportunity_penalty = 1.0 / (
        1.0
        + 0.08 * np.clip(opportunity_ratio - 1.0, 0.0, 6.0)
    )

    shared_neighbors = (
        long_mask.astype(float)
        @ long_mask.astype(float).T
    )

    shared_neighbor_signal = np.clip(
        shared_neighbors / max(float(long_k), 1.0),
        0.0,
        1.0,
    )

    concentration = long_mask.astype(float).sum(axis=0)
    concentration_reference = max(
        float(np.mean(concentration)),
        1.0,
    )

    concentration_ratio = np.clip(
        concentration / concentration_reference,
        0.0,
        3.0,
    )

    concentration_penalty = 1.0 / np.sqrt(
        (
            1.0
            + 0.10 * concentration_ratio[:, None]
        )
        * (
            1.0
            + 0.10 * concentration_ratio[None, :]
        )
    )

    scores = (
        distance_score
        * endpoint_affinity
        * mutual_preference
        * (0.52 + 0.48 * rank_score)
        * (1.0 + 0.28 * mutual_short)
        * (1.0 + 0.10 * mutual_long)
        * (1.0 + 0.24 * mutual_nearest)
        * (1.0 + 0.06 * shared_neighbor_signal)
        * scarcity_bonus
        * opportunity_penalty
        * concentration_penalty
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
