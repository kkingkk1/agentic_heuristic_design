import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, endpoint-aware edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(finite_positive, False)

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(distances[finite_positive]))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(distances), distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_offdiag = d[np.isfinite(d)]
    if finite_offdiag.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_offdiag))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    offdiag_values = d[~np.eye(n, dtype=bool)]
    positive_values = offdiag_values[offdiag_values > 0.0]

    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else fallback
    )
    global_scale = max(global_scale, eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 2))

    ranked_distances = np.sort(d, axis=1)
    nearest = ranked_distances[:, 0]
    local_neighbors = ranked_distances[:, :k]

    local_weights = np.linspace(1.0, 0.58, k, dtype=float)
    local_weights /= np.sum(local_weights)

    local_scale = local_neighbors @ local_weights
    local_scale = np.maximum(local_scale, eps)

    if n > 2:
        second_nearest = ranked_distances[:, 1]
    else:
        second_nearest = nearest.copy()

    local_gap = np.maximum(second_nearest - nearest, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = d / np.maximum(endpoint_scale, eps)

    distance_affinity = np.exp(
        -np.clip(normalized_distance ** 0.88, 0.0, 50.0)
    )

    endpoint_ratio_i = d / np.maximum(local_scale[:, None], eps)
    endpoint_ratio_j = d / np.maximum(local_scale[None, :], eps)

    endpoint_affinity_i = np.exp(
        -np.clip(endpoint_ratio_i ** 0.84, 0.0, 50.0)
    )
    endpoint_affinity_j = np.exp(
        -np.clip(endpoint_ratio_j ** 0.84, 0.0, 50.0)
    )

    bottleneck_affinity = np.minimum(
        endpoint_affinity_i,
        endpoint_affinity_j,
    )
    reciprocal_affinity = np.sqrt(
        endpoint_affinity_i * endpoint_affinity_j
    )

    row_order = np.argsort(d, axis=1, kind="stable")
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    normalized_rank_i = row_rank / max(float(n - 1), 1.0)
    normalized_rank_j = normalized_rank_i.T

    mutual_rank = 0.5 * (
        normalized_rank_i + normalized_rank_j
    )
    rank_affinity = np.exp(
        -1.32 * np.clip(mutual_rank, 0.0, 1.0)
    )

    candidate_width = max(float(k) * 1.15, 1.0)
    candidate_affinity_i = np.exp(
        -np.clip(row_rank / candidate_width, 0.0, 50.0)
    )
    candidate_affinity_j = np.exp(
        -np.clip(row_rank.T / candidate_width, 0.0, 50.0)
    )

    mutual_candidate_support = np.sqrt(
        candidate_affinity_i * candidate_affinity_j
    )
    candidate_gate = (
        0.58 + 0.42 * mutual_candidate_support
    )

    margin_i = np.clip(
        (second_nearest[:, None] - d)
        / np.maximum(
            second_nearest[:, None] + 1.15 * local_gap[:, None],
            eps,
        ),
        0.0,
        1.0,
    )
    margin_j = np.clip(
        (second_nearest[None, :] - d)
        / np.maximum(
            second_nearest[None, :] + 1.15 * local_gap[None, :],
            eps,
        ),
        0.0,
        1.0,
    )

    normalized_margin = np.sqrt(margin_i * margin_j)
    margin_bonus = 1.0 + 0.085 * normalized_margin

    density_ratio = np.sqrt(
        local_scale[:, None]
        / np.maximum(local_scale[None, :], eps)
    )
    density_balance = np.exp(
        -0.11 * np.abs(
            np.log(np.maximum(density_ratio, eps))
        )
    )

    global_guard = 0.95 + 0.05 * np.exp(
        -np.clip((d / global_scale) ** 0.86, 0.0, 50.0)
    )

    score = (
        distance_affinity
        * bottleneck_affinity
        * reciprocal_affinity
        * rank_affinity
        * candidate_gate
        * margin_bonus
        * density_balance
        * global_guard
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
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
