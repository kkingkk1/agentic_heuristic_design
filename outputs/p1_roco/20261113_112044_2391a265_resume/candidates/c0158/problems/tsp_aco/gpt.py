import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-priority scores for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distances)
    cleaned = np.where(finite, np.maximum(distances, 0.0), np.inf)
    np.fill_diagonal(cleaned, np.inf)

    transpose = cleaned.T
    forward = np.isfinite(cleaned)
    backward = np.isfinite(transpose)

    valid = forward | backward
    symmetric_distance = np.full((n, n), np.inf, dtype=float)

    both = forward & backward
    symmetric_distance[both] = 0.5 * (
        cleaned[both] + transpose[both]
    )
    symmetric_distance[forward & ~backward] = cleaned[forward & ~backward]
    symmetric_distance[backward & ~forward] = transpose[backward & ~forward]
    np.fill_diagonal(symmetric_distance, np.inf)

    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    finite_values = symmetric_distance[np.isfinite(symmetric_distance)]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    global_scale = max(global_scale, eps)

    neighbor_count = min(n - 1, max(3, int(np.ceil(np.sqrt(n)))))
    order = np.argsort(symmetric_distance, axis=1, kind="stable")

    ranks = np.zeros((n, n), dtype=float)
    local_scale = np.full(n, global_scale, dtype=float)
    nearest = np.full(n, -1, dtype=int)
    top_neighbors = np.zeros((n, n), dtype=bool)

    for i in range(n):
        candidates = order[i][np.isfinite(symmetric_distance[i, order[i]])]

        if candidates.size == 0:
            continue

        ranks[i, candidates] = 1.0 / np.arange(
            1, candidates.size + 1,
            dtype=float,
        )

        top = candidates[:neighbor_count]
        top_neighbors[i, top] = True
        nearest[i] = candidates[0]

        positive = symmetric_distance[i, candidates]
        positive = positive[positive > 0.0]

        if positive.size:
            local_scale[i] = float(
                np.median(positive[: min(5, positive.size)])
            )
        else:
            local_scale[i] = global_scale

    local_scale = np.maximum(local_scale, eps)
    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    normalized_distance = symmetric_distance / pair_scale
    normalized_distance = np.clip(normalized_distance, 0.0, 60.0)

    local_affinity = np.exp(-normalized_distance)
    global_affinity = 1.0 / (
        1.0 + np.clip(symmetric_distance / global_scale, 0.0, 60.0) ** 2
    )

    reciprocal_rank = np.sqrt(ranks * ranks.T)
    mutual_top = top_neighbors & top_neighbors.T

    mutual_nearest = np.zeros((n, n), dtype=bool)
    for i, j in enumerate(nearest):
        if j >= 0 and nearest[j] == i:
            mutual_nearest[i, j] = True
            mutual_nearest[j, i] = True

    endpoint_importance = np.maximum(ranks, ranks.T)

    isolation_ratio = pair_scale / (
        symmetric_distance + global_scale
    )
    isolation_ratio = np.clip(
        np.nan_to_num(
            isolation_ratio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )

    scores = (
        local_affinity
        * (0.70 + 0.30 * global_affinity)
        * (0.55 + 0.45 * reciprocal_rank)
        * (0.80 + 0.20 * endpoint_importance)
        * (1.0 + 0.35 * mutual_nearest.astype(float))
        * (1.0 + 0.15 * mutual_top.astype(float))
        * (1.0 + 0.15 * isolation_ratio)
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
