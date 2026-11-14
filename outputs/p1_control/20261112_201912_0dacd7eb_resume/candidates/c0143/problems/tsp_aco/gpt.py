import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, endpoint-aware edge-priority indicators for TSP."""
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
    eps = np.finfo(float).eps
    base_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(fallback, 1.0)
    )
    base_scale = max(base_scale, eps)

    ordered = np.sort(d, axis=1)
    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))

    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, eps)

    nearest = ordered[:, 0]
    second_nearest = ordered[:, 1] if n > 2 else nearest
    local_gap = np.maximum(second_nearest - nearest, eps)

    row_order = np.argsort(d, axis=1)
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)

    rank_i = row_rank / max(float(n - 1), 1.0)
    rank_j = rank_i.T

    geometric_affinity = np.exp(
        -np.clip(d / np.sqrt(local_scale[:, None] * local_scale[None, :]), 0.0, 40.0)
    )

    endpoint_affinity_i = np.exp(
        -np.clip(d / local_scale[:, None], 0.0, 40.0)
    )
    endpoint_affinity_j = np.exp(
        -np.clip(d / local_scale[None, :], 0.0, 40.0)
    )
    worst_endpoint_affinity = np.minimum(
        endpoint_affinity_i,
        endpoint_affinity_j,
    )

    mutuality = np.sqrt(np.maximum(geometric_affinity * worst_endpoint_affinity, 0.0))

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
    margin_bonus = 1.0 + 0.12 * np.sqrt(
        endpoint_margin_i * endpoint_margin_j
    )

    soft_candidate_weight = np.exp(
        -0.55 * np.clip((rank_i + rank_j) / max(float(k), 1.0), 0.0, 8.0)
    )

    nearest_index = row_order[:, 0]
    if n > 2:
        second_index = row_order[:, 1]
    else:
        second_index = nearest_index

    alternative_i = np.broadcast_to(nearest_index[:, None], (n, n)).copy()
    alternative_j = np.broadcast_to(nearest_index[None, :], (n, n)).copy()

    nearest_is_j = alternative_i == np.arange(n)[None, :]
    nearest_is_i = alternative_j == np.arange(n)[:, None]

    alternative_i[nearest_is_j] = np.broadcast_to(
        second_index[:, None],
        (n, n),
    )[nearest_is_j]
    alternative_j[nearest_is_i] = np.broadcast_to(
        second_index[None, :],
        (n, n),
    )[nearest_is_i]

    alternative_cross_distance = d[alternative_i, alternative_j]
    alternative_i_distance = d[np.arange(n)[:, None], alternative_i]
    alternative_j_distance = d[alternative_j, np.arange(n)[None, :]]

    two_opt_gain = (
        d
        + alternative_cross_distance
        - alternative_i_distance
        - alternative_j_distance
    )
    local_two_opt_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    two_opt_bonus = 1.0 + 0.10 * np.tanh(
        -two_opt_gain / np.maximum(local_two_opt_scale, eps)
    )

    score = (
        mutuality
        * soft_candidate_weight
        * margin_bonus
        * two_opt_bonus
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    return np.nan_to_num(
        np.clip(score, 0.0, 1.5),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
