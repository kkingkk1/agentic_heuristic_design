import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric normalized edge-priority indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0.0)
    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[finite_positive]
    fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.asarray(distances, dtype=float).copy()
    d = np.nan_to_num(
        d,
        nan=fallback,
        posinf=fallback,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d)
    d = np.where(valid, d, fallback)

    eps = np.finfo(float).eps
    off_diagonal = ~np.eye(n, dtype=bool)

    neighbor_count = max(
        1,
        min(n - 1, int(np.ceil(np.sqrt(n)))),
    )

    ordered = np.sort(d, axis=1)
    neighbor_distances = ordered[:, :neighbor_count]

    weights = np.linspace(
        1.0,
        0.55,
        neighbor_count,
        dtype=float,
    )
    local_scale = (
        neighbor_distances * weights[None, :]
    ).sum(axis=1) / weights.sum()
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = d / np.maximum(pair_scale, eps)

    closeness = np.exp(
        -np.clip(relative_distance ** 0.9, 0.0, 50.0)
    )

    inverse = 1.0 / np.maximum(d, eps)
    row_max = np.max(
        np.where(off_diagonal, inverse, 0.0),
        axis=1,
        keepdims=True,
    )
    normalized_inverse = np.divide(
        inverse,
        np.maximum(row_max, eps),
    )

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    normalized_rank = ranks / max(float(n - 1), 1.0)
    rank_score = np.exp(
        -0.9 * (
            normalized_rank
            + normalized_rank.T
        )
    )

    nearest_mask = (
        normalized_rank
        <= (neighbor_count - 1) / max(float(n - 1), 1.0)
    )
    nearest_mask &= off_diagonal
    mutual_support = nearest_mask & nearest_mask.T
    mutual_bonus = np.where(mutual_support, 0.12, 0.0)

    endpoint_score = np.sqrt(
        np.maximum(normalized_inverse, 0.0)
        * np.maximum(normalized_inverse.T, 0.0)
    )

    score = (
        0.45 * endpoint_score
        + 0.35 * closeness
        + 0.20 * rank_score
    )
    score *= 1.0 + mutual_bonus

    score = np.where(off_diagonal, score, 0.0)
    score = 0.5 * (score + score.T)
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
