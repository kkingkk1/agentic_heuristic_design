import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return endpoint-aware edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

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
    d = np.where(np.isfinite(d), d, fallback)

    positive_values = finite_values[finite_values > 0.0]
    scale_floor = np.finfo(float).eps
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(fallback, 1.0)
    )
    global_scale = max(global_scale, scale_floor)

    k = max(1, min(n - 1, int(np.sqrt(n)) + 2))
    ordered = np.sort(d, axis=1)

    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, scale_floor)

    nearest = ordered[:, 0]
    if n > 2:
        second_nearest = ordered[:, 1]
    else:
        second_nearest = ordered[:, 0]

    local_gap = np.maximum(second_nearest - nearest, 0.0)
    local_gap = np.maximum(local_gap, scale_floor)

    normalized = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    endpoint_affinity = np.exp(-np.clip(normalized, 0.0, 50.0))

    endpoint_ratio_i = np.exp(
        -np.clip(1.15 * d / local_scale[:, None], 0.0, 50.0)
    )
    endpoint_ratio_j = np.exp(
        -np.clip(1.15 * d / local_scale[None, :], 0.0, 50.0)
    )
    worst_endpoint_affinity = np.minimum(
        endpoint_ratio_i,
        endpoint_ratio_j,
    )

    row_order = np.argsort(d, axis=1)
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)

    normalized_rank = (
        row_rank / max(float(n - 1), 1.0)
        + row_rank.T / max(float(n - 1), 1.0)
    ) / 2.0
    rank_term = np.exp(-np.clip(0.85 * normalized_rank, 0.0, 1.0))

    endpoint_margin_i = np.clip(
        (second_nearest[:, None] - d) / local_gap[:, None],
        0.0,
        1.0,
    )
    endpoint_margin_j = np.clip(
        (second_nearest[None, :] - d) / local_gap[None, :],
        0.0,
        1.0,
    )
    margin_bonus = 1.0 + 0.10 * np.sqrt(
        endpoint_margin_i * endpoint_margin_j
    )

    mild_global_term = 0.90 + 0.10 * np.exp(
        -np.clip(d / global_scale, 0.0, 50.0)
    )

    score = (
        endpoint_affinity
        * worst_endpoint_affinity
        * rank_term
        * margin_bonus
        * mild_global_term
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
