import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, locally normalized edge-priority indicators for TSP."""

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

    positive_values = distances[finite_positive]
    fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(distances), distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    fallback = (
        float(np.max(finite_values))
        if finite_values.size
        else fallback
    )
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, fallback)

    eps = np.finfo(float).eps

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))
    ordered = np.sort(d, axis=1)
    neighbors = ordered[:, :k]

    neighbor_weights = np.linspace(1.0, 0.65, k, dtype=float)
    neighbor_weights /= np.sum(neighbor_weights)

    local_scale = neighbors @ neighbor_weights
    local_scale = np.maximum(local_scale, eps)

    nearest = ordered[:, 0]
    if n > 2:
        second_nearest = ordered[:, 1]
    else:
        second_nearest = nearest

    local_gap = np.maximum(second_nearest - nearest, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = d / np.maximum(pair_scale, eps)

    distance_affinity = np.exp(
        -np.clip(normalized_distance ** 0.9, 0.0, 50.0)
    )

    row_order = np.argsort(d, axis=1, kind="stable")
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    rank_scale = max(float(k), 1.0)
    rank_i = np.exp(
        -np.clip(row_rank / rank_scale, 0.0, 50.0)
    )
    rank_j = rank_i.T

    mutual_rank_support = np.sqrt(rank_i * rank_j)
    rank_affinity = 0.72 + 0.28 * mutual_rank_support

    margin_i = np.clip(
        (second_nearest[:, None] - d)
        / (second_nearest[:, None] + local_gap[:, None] + eps),
        0.0,
        1.0,
    )
    margin_j = np.clip(
        (second_nearest[None, :] - d)
        / (second_nearest[None, :] + local_gap[None, :] + eps),
        0.0,
        1.0,
    )

    reciprocal_margin = np.sqrt(margin_i * margin_j)
    margin_bonus = 1.0 + 0.12 * reciprocal_margin

    density_ratio = (
        local_scale[:, None]
        / np.maximum(local_scale[None, :], eps)
    )
    density_balance = np.exp(
        -0.18 * np.abs(np.log(np.maximum(density_ratio, eps)))
    )

    score = (
        distance_affinity
        * rank_affinity
        * margin_bonus
        * density_balance
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
