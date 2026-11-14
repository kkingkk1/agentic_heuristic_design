import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric endpoint-aware edge-priority indicators for TSP sampling."""
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

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, fallback)

    eps = np.finfo(float).eps
    positive_values = d[d > 0.0]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else fallback
    )
    global_scale = max(global_scale, eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 2))
    ordered = np.sort(d, axis=1)
    neighbors = ordered[:, :k]

    weights = np.linspace(1.0, 0.68, k, dtype=float)
    weights /= np.sum(weights)

    local_scale = neighbors @ weights
    local_scale = np.maximum(local_scale, eps)

    nearest = ordered[:, 0]
    second_nearest = ordered[:, 1] if n > 2 else nearest
    local_gap = np.maximum(second_nearest - nearest, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = d / np.maximum(pair_scale, eps)
    geometric_affinity = np.exp(
        -np.clip(relative_distance ** 0.84, 0.0, 50.0)
    )

    endpoint_ratio_i = d / local_scale[:, None]
    endpoint_ratio_j = d / local_scale[None, :]

    endpoint_affinity_i = np.exp(
        -np.clip(endpoint_ratio_i ** 0.78, 0.0, 50.0)
    )
    endpoint_affinity_j = np.exp(
        -np.clip(endpoint_ratio_j ** 0.78, 0.0, 50.0)
    )

    bottleneck_affinity = np.minimum(
        endpoint_affinity_i,
        endpoint_affinity_j,
    )
    compatibility = np.sqrt(
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

    rank_affinity = np.exp(
        -1.10 * np.clip(
            0.5 * (normalized_rank_i + normalized_rank_j),
            0.0,
            1.0,
        )
    )

    candidate_width = max(float(k) * 1.15, 1.0)
    candidate_i = np.exp(
        -np.clip(row_rank / candidate_width, 0.0, 50.0)
    )
    candidate_j = np.exp(
        -np.clip(row_rank.T / candidate_width, 0.0, 50.0)
    )
    mutual_candidate_support = np.sqrt(candidate_i * candidate_j)
    soft_candidate_gate = (
        0.60 + 0.40 * mutual_candidate_support
    )

    margin_i = np.clip(
        (second_nearest[:, None] - d)
        / (second_nearest[:, None] + 0.85 * local_gap[:, None] + eps),
        0.0,
        1.0,
    )
    margin_j = np.clip(
        (second_nearest[None, :] - d)
        / (second_nearest[None, :] + 0.85 * local_gap[None, :] + eps),
        0.0,
        1.0,
    )
    margin_bonus = 1.0 + 0.18 * np.sqrt(margin_i * margin_j)

    density_ratio = np.sqrt(
        local_scale[:, None] / np.maximum(local_scale[None, :], eps)
    )
    density_balance = np.exp(
        -0.20 * np.abs(np.log(np.maximum(density_ratio, eps)))
    )

    global_guard = 0.92 + 0.08 * np.exp(
        -np.clip((d / global_scale) ** 0.84, 0.0, 50.0)
    )

    reciprocal_reinforcement = (
        0.84 + 0.16 * compatibility
    )

    score = (
        geometric_affinity
        * bottleneck_affinity
        * compatibility
        * rank_affinity
        * soft_candidate_gate
        * margin_bonus
        * density_balance
        * global_guard
        * reciprocal_reinforcement
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
