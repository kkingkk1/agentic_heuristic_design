import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty_like(distance_matrix, dtype=float)
    if n == 1:
        return np.zeros_like(distance_matrix, dtype=float)

    d = np.nan_to_num(
        distance_matrix,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    ).copy()

    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros_like(d, dtype=float)

    fallback = max(float(np.max(finite_values)), 1.0)
    d[~np.isfinite(d)] = fallback * 10.0

    positive = d[d > 0]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    local_scale = np.partition(d, kth=min(2, n - 1), axis=1)[:, min(2, n - 1)]
    local_scale = np.maximum(local_scale, global_scale * 0.05)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    closeness = np.exp(-d / local_scale[:, None])

    order = np.argsort(d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = np.exp(-ranks / max(n - 1, 1))

    reverse_scale = np.maximum(local_scale[None, :], np.finfo(float).eps)
    reverse_closeness = np.exp(-d / reverse_scale)
    mutual_score = np.sqrt(closeness * reverse_closeness)

    score = (
        0.50 * closeness
        + 0.30 * rank_score
        + 0.20 * mutual_score
    )

    np.fill_diagonal(score, 0.0)

    maximum = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        np.maximum(maximum, np.finfo(float).eps),
        out=np.zeros_like(score),
        where=maximum > 0,
    )

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
