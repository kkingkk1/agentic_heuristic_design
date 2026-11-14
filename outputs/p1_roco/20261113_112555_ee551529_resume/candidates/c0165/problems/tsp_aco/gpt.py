import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    finite = np.isfinite(raw)
    nonnegative = np.maximum(raw, 0.0)

    forward = finite & ~diagonal
    backward = forward.T
    both = forward & backward

    distance = np.full((n, n), np.inf, dtype=float)

    distance[both] = 0.5 * (
        nonnegative[both] + nonnegative.T[both]
    )

    only_forward = forward & ~backward
    only_backward = backward & ~forward

    distance[only_forward] = nonnegative[only_forward]
    distance[only_backward] = nonnegative.T[only_backward]
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = distance[valid & (distance > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = distance[valid]
        global_scale = float(np.median(finite_values))

    global_scale = max(global_scale, eps)

    order = np.argsort(distance, axis=1, kind="stable")
    sorted_distance = np.take_along_axis(distance, order, axis=1)

    nearest = np.full(n, -1, dtype=int)
    second_nearest = np.full(n, global_scale, dtype=float)
    local_scale = np.full(n, global_scale, dtype=float)
    ranks = np.zeros((n, n), dtype=float)

    neighborhood_size = min(n - 1, max(2, int(np.ceil(np.sqrt(n)))))

    for i in range(n):
        candidates = order[i][np.isfinite(sorted_distance[i])]
        if candidates.size == 0:
            continue

        values = distance[i, candidates]
        nearest[i] = candidates[0]

        if candidates.size > 1:
            second_nearest[i] = values[1]
        else:
            second_nearest[i] = values[0]

        rank_values = np.arange(1, candidates.size + 1, dtype=float)
        ranks[i, candidates] = 1.0 / rank_values

        positive_values = values[values > 0.0]
        if positive_values.size:
            count = min(4, positive_values.size)
            local_scale[i] = float(np.median(positive_values[:count]))
        elif values.size:
            local_scale[i] = float(np.median(values[:min(4, values.size)]))

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

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        normalized_distance = distance / pair_scale
        distance_preference = np.exp(
            -np.clip(normalized_distance, 0.0, 60.0)
        )

        endpoint_preference = np.sqrt(
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

    top_mask = np.zeros((n, n), dtype=bool)
    nearest_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        candidates = order[i][np.isfinite(sorted_distance[i])]
        if candidates.size == 0:
            continue

        top_mask[i, candidates[:neighborhood_size]] = True
        nearest_mask[i, candidates[0]] = True

    mutual_top = top_mask & top_mask.T & valid
    mutual_nearest = nearest_mask & nearest_mask.T & valid

    scarcity = np.ones((n, n), dtype=float)

    for i in range(n):
        mask = valid[i]
        if not np.any(mask):
            continue

        gap = max(second_nearest[i], global_scale)
        ratio = gap / np.maximum(distance[i, mask], eps)

        scarcity[i, mask] += 0.25 * np.clip(
            ratio - 1.0,
            0.0,
            4.0,
        )

    scarcity = np.sqrt(scarcity * scarcity.T)

    scores = (
        distance_preference
        * endpoint_preference
        * (0.55 + 0.45 * reciprocal_rank)
        * (1.0 + 0.35 * mutual_top.astype(float))
        * (1.0 + 0.35 * mutual_nearest.astype(float))
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
