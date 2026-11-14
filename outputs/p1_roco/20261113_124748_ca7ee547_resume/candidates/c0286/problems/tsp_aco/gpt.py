import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw) & (raw >= 0.0) & ~diagonal

    forward = finite
    backward = finite.T
    valid = forward | backward

    distance = np.full((n, n), np.inf, dtype=float)
    both = forward & backward

    distance[both] = 0.5 * (raw[both] + raw.T[both])
    distance[forward & ~backward] = raw[forward & ~backward]
    distance[backward & ~forward] = raw.T[backward & ~forward]
    np.fill_diagonal(distance, np.inf)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    order = np.argsort(distance, axis=1, kind="stable")
    sorted_distance = np.take_along_axis(distance, order, axis=1)
    finite_sorted = np.isfinite(sorted_distance)

    ranks = np.zeros((n, n), dtype=float)
    top_neighbors = np.zeros((n, n), dtype=bool)
    nearest = np.full(n, -1, dtype=int)
    local_scale = np.full(n, global_scale, dtype=float)
    second_nearest = np.full(n, global_scale, dtype=float)

    neighborhood_size = min(max(3, int(np.ceil(np.sqrt(n)))), n - 1)

    for i in range(n):
        neighbors = order[i][finite_sorted[i]]
        if neighbors.size == 0:
            continue

        values = distance[i, neighbors]
        nearest[i] = int(neighbors[0])

        if values.size > 1:
            second_nearest[i] = float(values[1])
        else:
            second_nearest[i] = float(values[0])

        positive = values[values > 0.0]
        reference = positive[: min(5, positive.size)]
        if reference.size:
            local_scale[i] = float(np.median(reference))
        else:
            local_scale[i] = float(np.median(values[: min(5, values.size)]))

        rank_values = np.arange(1, neighbors.size + 1, dtype=float)
        ranks[i, neighbors] = 1.0 / rank_values
        top_neighbors[i, neighbors[:neighborhood_size]] = True

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative_distance = distance / pair_scale
        endpoint_relative = np.sqrt(
            np.maximum(distance / local_scale[:, None], 0.0)
            * np.maximum(distance / local_scale[None, :], 0.0)
        )

    positive_relative = relative_distance[valid & (relative_distance > 0.0)]
    if positive_relative.size:
        temperature = float(np.median(positive_relative))
    else:
        temperature = 1.0

    temperature = float(np.clip(temperature, 0.25, 2.5))

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        distance_score = np.exp(
            -np.clip(relative_distance / temperature, 0.0, 60.0)
        )
        endpoint_score = np.exp(
            -np.clip(endpoint_relative, 0.0, 60.0)
        )

    rank_consensus = np.sqrt(ranks * ranks.T)
    rank_strength = 0.55 + 0.45 * rank_consensus

    mutual_local = top_neighbors & top_neighbors.T & valid

    node_ids = np.arange(n)
    mutual_nearest = (
        valid
        & (nearest[:, None] == node_ids[None, :])
        & (nearest[None, :] == node_ids[:, None])
    )

    endpoint_best = np.minimum(local_scale[:, None], local_scale[None, :])

    with np.errstate(divide="ignore", invalid="ignore"):
        opportunity_ratio = distance / np.maximum(endpoint_best, eps)
        regret_i = np.maximum(
            second_nearest[:, None] / np.maximum(distance, eps) - 1.0,
            0.0,
        )
        regret_j = np.maximum(
            second_nearest[None, :] / np.maximum(distance, eps) - 1.0,
            0.0,
        )

    competition_penalty = 1.0 / (
        1.0 + 0.22 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    regret_bonus = 1.0 + 0.16 * np.clip(
        np.sqrt(regret_i * regret_j),
        0.0,
        4.0,
    )

    local_density = np.sqrt(
        np.maximum(
            global_scale / np.maximum(local_scale[:, None], eps),
            0.0,
        )
        * np.maximum(
            global_scale / np.maximum(local_scale[None, :], eps),
            0.0,
        )
    )
    sparsity_bonus = 1.0 + 0.10 * np.clip(local_density - 1.0, 0.0, 3.0)

    scores = (
        distance_score
        * endpoint_score
        * rank_strength
        * (1.0 + 0.30 * mutual_local.astype(float))
        * (1.0 + 0.42 * mutual_nearest.astype(float))
        * competition_penalty
        * regret_bonus
        * sparsity_bonus
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
