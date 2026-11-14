import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, sparse, endpoint-aware edge-priority indicators."""
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
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))

    ordered = np.sort(d, axis=1)
    neighbors = ordered[:, :k]

    weights = np.linspace(1.0, 0.65, k, dtype=float)
    weights /= np.sum(weights)

    local_scale = neighbors @ weights
    local_scale = np.maximum(local_scale, eps)

    nearest = ordered[:, 0]
    if n > 2:
        second_nearest = ordered[:, 1]
    else:
        second_nearest = nearest

    row_order = np.argsort(d, axis=1, kind="stable")
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    normalized_rank = row_rank / max(float(n - 1), 1.0)
    rank_affinity = np.exp(
        -1.25 * 0.5 * (normalized_rank + normalized_rank.T)
    )

    candidate_width = max(float(k), 1.0)
    candidate_i = np.exp(
        -np.clip(row_rank / candidate_width, 0.0, 40.0)
    )
    candidate_j = candidate_i.T
    mutual_support = np.sqrt(candidate_i * candidate_j)
    candidate_gate = 0.35 + 0.65 * mutual_support

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    distance_affinity = np.exp(
        -np.clip(
            np.sqrt(d / np.maximum(pair_scale, eps)),
            0.0,
            40.0,
        )
    )

    endpoint_ratio_i = d / np.maximum(local_scale[:, None], eps)
    endpoint_ratio_j = d / np.maximum(local_scale[None, :], eps)
    endpoint_affinity = np.exp(
        -0.5
        * np.clip(endpoint_ratio_i + endpoint_ratio_j, 0.0, 80.0)
    )

    gap = np.maximum(second_nearest - nearest, eps)
    regret_i = np.maximum(d - second_nearest[:, None], 0.0) / gap[:, None]
    regret_j = np.maximum(d - second_nearest[None, :], 0.0) / gap[None, :]
    regret_penalty = np.exp(
        -0.12 * np.clip(regret_i + regret_j, 0.0, 80.0)
    )

    score = (
        distance_affinity
        * endpoint_affinity
        * rank_affinity
        * candidate_gate
        * regret_penalty
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
