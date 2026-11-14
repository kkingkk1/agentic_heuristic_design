import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback):
        fallback = 1.0
    d = np.where(np.isfinite(d), d, fallback)

    positive_values = finite_values[finite_values > 0.0]
    if positive_values.size:
        global_scale = float(np.percentile(positive_values, 55.0))
    else:
        global_scale = max(fallback, 1.0)
    global_scale = max(global_scale, np.finfo(float).eps)

    k = max(2, min(n - 1, int(np.sqrt(n)) + 2))
    ordered = np.sort(d, axis=1)

    neighbor_distances = ordered[:, :k]
    weights = np.linspace(1.0, 0.55, k, dtype=float)
    local_scale = np.sum(neighbor_distances * weights[None, :], axis=1)
    local_scale /= float(np.sum(weights))
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    local_pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = d / local_pair_scale

    distance_term = np.exp(-np.clip(relative_distance ** 0.85, 0.0, 50.0))

    row_order = np.argsort(d, axis=1, kind="stable")
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    normalized_rank = row_rank / max(float(n - 1), 1.0)
    rank_term = (
        1.0
        + 0.9 * normalized_rank
        + 0.9 * normalized_rank.T
        + 0.35 * normalized_rank * normalized_rank.T
    ) ** -1.15

    endpoint_ratio_i = d / local_scale[:, None]
    endpoint_ratio_j = d / local_scale[None, :]
    compatibility_penalty = (
        np.log1p(endpoint_ratio_i) ** 1.25
        + np.log1p(endpoint_ratio_j) ** 1.25
    )
    local_compatibility = np.exp(
        -np.clip(compatibility_penalty, 0.0, 50.0)
    )

    global_term = np.exp(
        -np.clip((d / global_scale) ** 0.9, 0.0, 50.0)
    )
    global_adjustment = 0.35 + 0.65 * global_term

    score = (
        distance_term ** 1.15
        * rank_term
        * local_compatibility ** 1.2
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
    )
