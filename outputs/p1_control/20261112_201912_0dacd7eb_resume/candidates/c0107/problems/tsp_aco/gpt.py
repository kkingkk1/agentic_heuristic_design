import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d)
    finite_values = d[finite]

    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback):
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)

    positive = d[(d > 0.0) & np.isfinite(d)]
    global_scale = (
        float(np.percentile(positive, 50.0))
        if positive.size
        else max(fallback, 1.0)
    )
    global_scale = max(global_scale, eps)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))
    ordered = np.sort(work, axis=1)

    neighbor_distances = ordered[:, :k]
    weights = np.linspace(1.0, 0.6, k, dtype=float)
    local_scale = (
        neighbor_distances * weights[None, :]
    ).sum(axis=1) / weights.sum()
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = d / (pair_scale + eps)

    distance_term = np.exp(
        -np.clip(relative_distance ** 0.9, 0.0, 50.0)
    )

    row_order = np.argsort(work, axis=1, kind="mergesort")
    row_rank = np.empty_like(row_order, dtype=float)
    row_index = np.arange(n)[:, None]
    row_rank[row_index, row_order] = np.arange(n, dtype=float)[None, :]

    rank_scale = max(float(n - 1), 1.0)
    normalized_rank = row_rank / rank_scale
    mutual_rank = np.sqrt(
        (1.0 / (1.0 + normalized_rank))
        * (1.0 / (1.0 + normalized_rank.T))
    )

    endpoint_ratio_i = d / (local_scale[:, None] + eps)
    endpoint_ratio_j = d / (local_scale[None, :] + eps)

    endpoint_affinity_i = np.exp(
        -np.clip(endpoint_ratio_i ** 0.8, 0.0, 50.0)
    )
    endpoint_affinity_j = np.exp(
        -np.clip(endpoint_ratio_j ** 0.8, 0.0, 50.0)
    )
    weakest_endpoint = np.minimum(
        endpoint_affinity_i,
        endpoint_affinity_j,
    )

    global_term = np.exp(
        -np.clip((d / global_scale) ** 0.85, 0.0, 50.0)
    )
    global_adjustment = 0.45 + 0.55 * global_term

    score = (
        distance_term ** 1.15
        * mutual_rank ** 0.9
        * weakest_endpoint ** 0.8
        * global_adjustment
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
