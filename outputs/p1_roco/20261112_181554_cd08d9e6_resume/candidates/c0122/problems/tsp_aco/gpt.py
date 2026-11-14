import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros_like(distances, dtype=float)

    d = 0.5 * (distances + distances.T)
    valid = np.isfinite(d) & (d > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid]
    global_scale = float(np.median(positive))
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = float(np.max(positive))
    global_scale = max(global_scale, np.finfo(float).eps)

    fallback = float(np.max(positive))
    working = np.where(valid, d, fallback)
    np.fill_diagonal(working, fallback)

    k = min(7, n - 1)
    nearest_values = np.partition(working, kth=k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest_values, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    relative_distance = working / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance /= max(global_scale / global_scale, np.finfo(float).eps)

    inverse_distance = np.where(
        valid,
        1.0 / np.maximum(relative_distance, np.finfo(float).eps) ** 0.9,
        0.0,
    )

    order = np.argsort(working, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_decay = max(1.0, 0.8 * np.sqrt(float(n)))
    rank_term = np.exp(-ranks / rank_decay)
    rank_term *= valid

    nearest = np.argmin(working, axis=1)
    has_neighbor = np.any(valid, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    for i in range(n):
        if not has_neighbor[i]:
            continue
        j = int(nearest[i])
        if valid[i, j] and has_neighbor[j] and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    proximity = 1.0 / (1.0 + relative_distance ** 0.9)
    long_edge_penalty = np.exp(
        -0.55 * np.maximum(relative_distance - 1.0, 0.0)
    )

    score = (
        inverse_distance
        * proximity
        * long_edge_penalty
        * (0.50 + 0.50 * np.sqrt(rank_term * rank_term.T))
        * (1.0 + 0.45 * mutual_nearest)
    )

    score *= valid
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0,
    )

    score = 0.5 * (score + score.T)
    score *= valid
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
