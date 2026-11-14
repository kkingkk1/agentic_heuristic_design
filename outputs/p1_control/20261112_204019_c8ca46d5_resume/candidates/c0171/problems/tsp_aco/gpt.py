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

    off_diagonal = ~np.eye(n, dtype=bool)
    finite_values = d[off_diagonal & np.isfinite(d)]

    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(finite_values, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(finite_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)

    positive_values = finite_values[finite_values > 0.0]
    if positive_values.size == 0:
        score = np.ones((n, n), dtype=float)
        np.fill_diagonal(score, 0.0)
        return score

    eps = np.finfo(float).eps

    global_scale = float(np.percentile(positive_values, 60.0))
    global_scale = max(global_scale, eps)

    k = int(np.sqrt(n)) + 1
    k = max(2, min(n - 1, k))

    ordered = np.sort(d, axis=1)
    neighbor_distances = ordered[:, :k]

    weights = np.linspace(1.0, 0.45, k, dtype=float)
    weights /= np.sum(weights)

    local_scale = neighbor_distances @ weights
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = d / np.maximum(pair_scale, eps)

    distance_term = np.exp(
        -np.clip(relative_distance ** 0.82, 0.0, 50.0)
    )

    row_order = np.argsort(d, axis=1, kind="stable")
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    rank_scale = max(float(n - 1), 1.0)
    rank_i = row_rank / rank_scale
    rank_j = rank_i.T

    rank_term = np.exp(
        -0.9 * (rank_i ** 0.85 + rank_j ** 0.85)
    )

    candidate_mask = (row_rank < float(k)) & (row_rank.T < float(k))
    mutual_bonus = np.where(candidate_mask, 1.18, 1.0)

    endpoint_ratio_i = d / np.maximum(local_scale[:, None], eps)
    endpoint_ratio_j = d / np.maximum(local_scale[None, :], eps)

    compatibility = np.exp(
        -0.5
        * (
            np.log1p(np.clip(endpoint_ratio_i, 0.0, None)) ** 1.25
            + np.log1p(np.clip(endpoint_ratio_j, 0.0, None)) ** 1.25
        )
    )

    global_term = np.exp(
        -np.clip((d / global_scale) ** 0.72, 0.0, 50.0)
    )
    global_adjustment = 0.45 + 0.55 * global_term

    score = (
        distance_term ** 1.35
        * rank_term
        * compatibility ** 1.15
        * global_adjustment
        * mutual_bonus
    )

    score = 0.5 * (score + score.T)
    score[~off_diagonal] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
