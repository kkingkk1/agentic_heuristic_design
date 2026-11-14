import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust, symmetric edge-priority indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite_positive = np.isfinite(distances) & (distances > 0.0)
    finite_positive &= off_diagonal

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(distances[finite_positive]))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(distances), distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    eps = np.finfo(float).eps
    positive_values = d[off_diagonal & (d > 0.0)]

    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    global_scale = max(float(np.median(positive_values)), eps)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))

    order = np.argsort(work, axis=1, kind="stable")
    neighbors = np.take_along_axis(
        work,
        order[:, :k],
        axis=1,
    )

    weights = np.linspace(1.0, 0.55, k, dtype=float)
    weights /= np.sum(weights)

    local_scale = neighbors @ weights
    local_scale = np.maximum(local_scale, eps)

    alternative = np.maximum(neighbors[:, -1], eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = d / np.maximum(pair_scale, eps)
    geometric_affinity = np.exp(
        -np.clip(relative_distance ** 0.88, 0.0, 50.0)
    )

    endpoint_ratio_i = d / local_scale[:, None]
    endpoint_ratio_j = d / local_scale[None, :]

    endpoint_i = np.exp(
        -np.clip(endpoint_ratio_i ** 0.85, 0.0, 50.0)
    )
    endpoint_j = np.exp(
        -np.clip(endpoint_ratio_j ** 0.85, 0.0, 50.0)
    )

    bottleneck_affinity = np.minimum(endpoint_i, endpoint_j)
    balanced_affinity = np.sqrt(endpoint_i * endpoint_j)

    ranks = np.empty((n, n), dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)[None, :]

    normalized_rank = ranks / max(float(n - 1), 1.0)
    reciprocal_rank = np.exp(
        -1.35
        * np.clip(
            0.5 * (normalized_rank + normalized_rank.T),
            0.0,
            1.0,
        )
    )

    candidate_scale = max(float(k), 1.0)
    candidate_i = np.exp(
        -np.clip(ranks / candidate_scale, 0.0, 50.0)
    )
    candidate_j = np.exp(
        -np.clip(ranks.T / candidate_scale, 0.0, 50.0)
    )

    mutual_candidate_support = np.sqrt(
        candidate_i * candidate_j
    )
    candidate_gate = (
        0.58 + 0.42 * mutual_candidate_support
    )

    margin_i = np.clip(
        (alternative[:, None] - d)
        / (alternative[:, None] + local_scale[:, None] + eps),
        0.0,
        1.0,
    )
    margin_j = np.clip(
        (alternative[None, :] - d)
        / (alternative[None, :] + local_scale[None, :] + eps),
        0.0,
        1.0,
    )

    margin_bonus = 1.0 + 0.10 * np.sqrt(margin_i * margin_j)

    density_ratio = np.sqrt(
        local_scale[:, None]
        / np.maximum(local_scale[None, :], eps)
    )
    density_balance = np.exp(
        -0.25
        * np.abs(np.log(np.maximum(density_ratio, eps)))
    )

    global_affinity = np.exp(
        -np.clip((d / global_scale) ** 0.80, 0.0, 50.0)
    )
    global_guard = 0.88 + 0.12 * global_affinity

    score = (
        geometric_affinity
        * bottleneck_affinity
        * balanced_affinity
        * reciprocal_rank
        * candidate_gate
        * margin_bonus
        * density_balance
        * global_guard
    )

    score = 0.5 * (score + score.T)
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(score, 0.0)

    values = score[off_diagonal]
    if values.size == 0:
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(values))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return np.clip(score, 0.0, 1.0).astype(float, copy=False)
