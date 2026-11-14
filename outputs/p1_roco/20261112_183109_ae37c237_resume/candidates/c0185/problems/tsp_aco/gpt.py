import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0)
    finite_values = distances[finite_positive]

    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.median(finite_values))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    d = np.where(finite_positive, distances, fallback)
    d = 0.5 * (d + d.T)
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(d, np.inf)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    global_scale = float(np.median(d[valid]))
    global_scale = max(global_scale, np.finfo(float).eps)

    k = min(6, n - 1)
    nearest = np.partition(d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = d / geometric_scale

    visibility = geometric_scale / np.maximum(d, np.finfo(float).eps)
    proximity = np.exp(-0.95 * relative_distance)
    long_edge_penalty = np.exp(
        -0.68 * np.maximum(relative_distance - 1.0, 0.0)
    )

    order = np.argsort(d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_strength = np.exp(
        -ranks / max(1.0, np.sqrt(float(n)) * 0.85)
    )
    bidirectional_rank = np.sqrt(rank_strength * rank_strength.T)

    nearest_neighbor = order[:, 0]
    mutual_neighbor = (
        nearest_neighbor[:, None] == np.arange(n)[None, :]
    ) & (
        nearest_neighbor[None, :] == np.arange(n)[:, None]
    )

    accessibility = np.exp(-d / local_scale[:, None])
    ambiguity = np.sum(
        np.where(valid, accessibility, 0.0),
        axis=1,
    )
    scarcity = 1.0 / np.sqrt(
        1.0 + ambiguity[:, None] + ambiguity[None, :]
    )

    score = (
        visibility
        * (0.46 + 0.54 * proximity)
        * (0.57 + 0.43 * bidirectional_rank)
        * long_edge_penalty
        * (1.0 + 0.44 * scarcity)
    )

    score *= np.where(mutual_neighbor, 1.85, 1.0)
    score *= valid

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
