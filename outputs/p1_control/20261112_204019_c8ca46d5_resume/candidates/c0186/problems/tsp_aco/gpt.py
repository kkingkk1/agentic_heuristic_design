import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_input = np.isfinite(distances)
    positive_input = finite_input & (distances > 0.0)
    np.fill_diagonal(positive_input, False)

    if not np.any(positive_input):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[positive_input]
    fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(distances), distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d)
    finite_values = d[valid]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)

    eps = np.finfo(float).eps
    positive_values = d[d > 0.0]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(fallback, 1.0)
    )
    global_scale = max(global_scale, eps)

    neighbor_count = max(1, min(n - 1, int(np.sqrt(n)) + 2))
    ordered = np.sort(d, axis=1)
    local_neighbors = ordered[:, :neighbor_count]

    weights = np.linspace(1.0, 0.72, neighbor_count)
    local_scale = local_neighbors @ weights / np.sum(weights)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = d / np.maximum(pair_scale, eps)

    base_affinity = np.exp(
        -np.clip(relative_distance ** 0.84, 0.0, 50.0)
    )

    endpoint_affinity_i = np.exp(
        -np.clip(0.92 * d / local_scale[:, None], 0.0, 50.0)
    )
    endpoint_affinity_j = np.exp(
        -np.clip(0.92 * d / local_scale[None, :], 0.0, 50.0)
    )
    endpoint_affinity = np.sqrt(
        endpoint_affinity_i * endpoint_affinity_j
    )

    row_order = np.argsort(d, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    rank_scale = max(float(n - 1), 1.0)
    bidirectional_rank = (
        row_rank / rank_scale
        + row_rank.T / rank_scale
    ) / 2.0
    rank_term = np.exp(
        -np.clip(0.68 * bidirectional_rank, 0.0, 50.0)
    )

    nearest = ordered[:, 0]
    second_nearest = (
        ordered[:, 1] if n > 2 else ordered[:, 0]
    )
    gap = np.maximum(second_nearest - nearest, eps)

    advantage_i = np.clip(
        (second_nearest[:, None] - d) / gap[:, None],
        0.0,
        1.0,
    )
    advantage_j = np.clip(
        (second_nearest[None, :] - d) / gap[None, :],
        0.0,
        1.0,
    )
    mutual_advantage = np.sqrt(advantage_i * advantage_j)
    advantage_bonus = 1.0 + 0.24 * mutual_advantage

    global_term = 0.90 + 0.10 * np.exp(
        -np.clip(d / global_scale, 0.0, 50.0)
    )

    score = (
        base_affinity
        * endpoint_affinity
        * rank_term
        * advantage_bonus
        * global_term
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
