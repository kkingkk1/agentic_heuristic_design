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

    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)

    positive = finite[finite > 0.0]
    if positive.size:
        global_scale = float(np.percentile(positive, 60.0))
    else:
        global_scale = fallback
    global_scale = max(global_scale, eps)

    k = min(n - 1, max(2, int(np.sqrt(n)) + 1))

    order = np.argsort(d, axis=1, kind="stable")
    nearest = np.take_along_axis(d, order[:, :k], axis=1)

    weights = np.linspace(1.0, 0.45, k, dtype=float)
    local_scale = (
        np.sum(nearest * weights[None, :], axis=1)
        / np.sum(weights)
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = d / np.maximum(pair_scale, eps)

    distance_term = np.exp(
        -np.clip(relative_distance ** 0.85, 0.0, 50.0)
    )

    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    rank_denominator = max(float(n - 1), 1.0)
    normalized_rank = row_rank / rank_denominator

    bidirectional_rank = 0.5 * (
        normalized_rank + normalized_rank.T
    )
    rank_term = np.exp(-1.35 * bidirectional_rank)

    mutuality = np.exp(
        -0.8
        * np.clip(
            (row_rank + row_rank.T) / max(float(2 * k), 1.0),
            0.0,
            50.0,
        )
    )
    mutuality_term = 0.72 + 0.48 * mutuality

    endpoint_ratio_i = d / np.maximum(local_scale[:, None], eps)
    endpoint_ratio_j = d / np.maximum(local_scale[None, :], eps)

    local_penalty = np.log1p(endpoint_ratio_i) + np.log1p(endpoint_ratio_j)
    compatibility_term = np.exp(
        -np.clip(0.75 * local_penalty, 0.0, 50.0)
    )

    global_term = np.exp(
        -np.clip((d / global_scale) ** 0.8, 0.0, 50.0)
    )
    global_adjustment = 0.55 + 0.45 * global_term

    score = (
        distance_term ** 1.35
        * rank_term
        * mutuality_term
        * compatibility_term ** 1.15
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
