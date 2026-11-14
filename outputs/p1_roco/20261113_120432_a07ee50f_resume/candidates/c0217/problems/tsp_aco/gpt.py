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
    both = forward & backward

    distance = np.full((n, n), np.inf, dtype=float)

    distance[both] = 0.5 * (
        raw[both] + raw.T[both]
    )
    distance[forward & ~backward] = raw[forward & ~backward]
    distance[backward & ~forward] = raw.T[backward & ~forward]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance)

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

    local_scale = np.full(n, global_scale, dtype=float)
    nearest = np.full(n, -1, dtype=int)
    second_nearest = np.full(n, global_scale, dtype=float)

    neighborhood_size = min(max(3, int(np.ceil(np.sqrt(n)))), n - 1)

    for i in range(n):
        neighbors = order[i][np.isfinite(sorted_distance[i])]
        if neighbors.size == 0:
            continue

        values = distance[i, neighbors]
        nearest[i] = neighbors[0]

        if values.size > 1:
            second_nearest[i] = values[1]
        else:
            second_nearest[i] = values[0]

        positive = values[values > 0.0]
        if positive.size:
            local_scale[i] = float(
                np.median(positive[: min(4, positive.size)])
            )
        else:
            local_scale[i] = float(
                np.median(values[: min(4, values.size)])
            )

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)

    pair_scale = np.sqrt(
        np.maximum(
            local_scale[:, None] * local_scale[None, :],
            eps,
        )
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative_distance = distance / pair_scale
        distance_score = np.exp(
            -np.clip(relative_distance, 0.0, 50.0)
        )

        endpoint_score = np.sqrt(
            np.exp(
                -np.clip(
                    distance / local_scale[:, None],
                    0.0,
                    50.0,
                )
            )
            * np.exp(
                -np.clip(
                    distance / local_scale[None, :],
                    0.0,
                    50.0,
                )
            )
        )

    ranks = np.zeros((n, n), dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)

    for i in range(n):
        neighbors = order[i][np.isfinite(sorted_distance[i])]
        ranks[i, neighbors] = 1.0 / rank_values[: neighbors.size]

    rank_consensus = np.sqrt(ranks * ranks.T)

    top_neighbors = np.zeros((n, n), dtype=bool)
    for i in range(n):
        neighbors = order[i][np.isfinite(sorted_distance[i])]
        top_neighbors[i, neighbors[:neighborhood_size]] = True

    mutual_local = top_neighbors & top_neighbors.T & valid

    mutual_nearest = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    )

    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        opportunity_ratio = distance / np.maximum(endpoint_best, eps)

    competition_penalty = 1.0 / (
        1.0
        + 0.18 * np.clip(opportunity_ratio - 1.0, 0.0, 4.0)
    )

    gap = np.maximum(
        second_nearest[:, None],
        second_nearest[None, :],
    )

    scarcity = 1.0 + 0.08 * np.clip(
        gap / np.maximum(distance, eps) - 1.0,
        0.0,
        3.0,
    )

    scores = (
        distance_score
        * endpoint_score
        * (0.58 + 0.42 * rank_consensus)
        * (1.0 + 0.28 * mutual_local.astype(float))
        * (1.0 + 0.38 * mutual_nearest.astype(float))
        * competition_penalty
        * scarcity
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
