import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    raw = 0.5 * (distances + distances.T)
    off_diagonal = ~np.eye(n, dtype=bool)

    finite_positive = raw[off_diagonal & np.isfinite(raw) & (raw > 0.0)]
    if finite_positive.size == 0:
        finite_nonnegative = raw[off_diagonal & np.isfinite(raw) & (raw >= 0.0)]
        if finite_nonnegative.size == 0:
            return np.zeros((n, n), dtype=float)
        fallback = 1.0
    else:
        fallback = float(np.max(finite_positive))
        if not np.isfinite(fallback) or fallback <= 0.0:
            fallback = 1.0

    d = np.where(np.isfinite(raw) & (raw >= 0.0), raw, fallback)
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite_edges = d[np.isfinite(d)]
    if finite_edges.size == 0:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_edges = finite_edges[finite_edges > 0.0]
    global_scale = (
        float(np.median(positive_edges))
        if positive_edges.size
        else max(fallback, 1.0)
    )
    global_scale = max(global_scale, eps)

    ordered = np.sort(d, axis=1)
    neighbor_count = max(1, min(n - 1, int(np.sqrt(n)) + 2))
    neighbors = ordered[:, :neighbor_count]

    weights = np.linspace(1.0, 0.55, neighbor_count)
    weights /= np.sum(weights)
    local_scale = neighbors @ weights

    if np.any(local_scale <= eps):
        positive_local = local_scale[local_scale > eps]
        replacement = (
            float(np.median(positive_local))
            if positive_local.size
            else global_scale
        )
        local_scale = np.where(local_scale > eps, local_scale, replacement)

    local_scale = np.maximum(local_scale, eps)
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    relative_distance = d / np.maximum(pair_scale, eps)
    local_affinity = np.exp(
        -np.clip(relative_distance ** 0.78, 0.0, 50.0)
    )

    endpoint_affinity = np.exp(
        -0.5 * np.clip(d / local_scale[:, None], 0.0, 50.0)
        -0.5 * np.clip(d / local_scale[None, :], 0.0, 50.0)
    )

    row_order = np.argsort(d, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    rank_denominator = max(n - 2, 1)
    mean_rank = 0.5 * (row_rank + row_rank.T) / rank_denominator
    rank_affinity = np.exp(
        -np.clip(0.62 * mean_rank ** 0.85, 0.0, 50.0)
    )

    nearest = ordered[:, 0]
    if n > 2:
        second_nearest = ordered[:, 1]
    else:
        second_nearest = nearest.copy()

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
    advantage_bonus = 1.0 + 0.24 * mutual_advantage ** 1.35

    global_term = 0.90 + 0.10 * np.exp(
        -np.clip(d / global_scale, 0.0, 50.0)
    )

    score = (
        local_affinity
        * endpoint_affinity
        * rank_affinity
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
