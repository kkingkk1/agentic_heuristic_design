import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0) & ~diagonal

    forward = finite_nonnegative
    backward = finite_nonnegative.T
    both = forward & backward

    distance = np.full((n, n), np.inf, dtype=float)

    distance[both] = 0.5 * (
        np.maximum(raw[both], 0.0)
        + np.maximum(raw.T[both], 0.0)
    )

    only_forward = forward & ~backward
    only_backward = backward & ~forward

    distance[only_forward] = np.maximum(raw[only_forward], 0.0)
    distance[only_backward] = np.maximum(raw.T[only_backward], 0.0)
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    local_scale = np.full(n, global_scale, dtype=float)
    nearest = np.full(n, -1, dtype=int)
    second_nearest = np.full(n, global_scale, dtype=float)
    ranks = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        neighbors = neighbors[
            np.argsort(distance[i, neighbors], kind="stable")
        ]
        values = distance[i, neighbors]

        nearest[i] = int(neighbors[0])
        second_nearest[i] = (
            float(values[1]) if values.size > 1 else float(values[0])
        )

        count = min(4, values.size)
        local_scale[i] = float(np.mean(values[:count]))

        ranks[i, neighbors] = 1.0 / np.arange(
            1.0, values.size + 1.0
        )

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        np.maximum(
            local_scale[:, None] * local_scale[None, :],
            eps,
        )
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative_distance = distance / pair_scale
        shortness = np.exp(
            -np.clip(relative_distance, 0.0, 60.0)
        )

        endpoint_affinity = np.sqrt(
            np.exp(
                -np.clip(
                    distance / local_scale[:, None],
                    0.0,
                    60.0,
                )
            )
            * np.exp(
                -np.clip(
                    distance / local_scale[None, :],
                    0.0,
                    60.0,
                )
            )
        )

    reciprocal_rank = np.sqrt(ranks * ranks.T)

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if j >= 0 and nearest[j] == i and valid[i, j]:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    top_k = min(3, n - 1)
    mutual_top = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        neighbors = neighbors[
            np.argsort(distance[i, neighbors], kind="stable")
        ]
        selected = neighbors[:top_k]
        mutual_top[i, selected] = 1.0

    mutual_top *= mutual_top.T
    mutual_top *= valid

    with np.errstate(divide="ignore", invalid="ignore"):
        scarcity = second_nearest[:, None] / np.maximum(
            distance,
            eps,
        )

    scarcity = np.clip(scarcity - 1.0, 0.0, 4.0)
    scarcity = 1.0 + 0.12 * np.sqrt(
        scarcity * scarcity.T
    )

    scores = (
        shortness
        * endpoint_affinity
        * (0.50 + 0.50 * reciprocal_rank)
        * (1.0 + 0.32 * mutual_nearest)
        * (1.0 + 0.18 * mutual_top)
        * scarcity
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    scores = np.maximum(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
