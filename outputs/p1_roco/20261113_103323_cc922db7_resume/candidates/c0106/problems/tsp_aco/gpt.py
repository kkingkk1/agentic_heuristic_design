import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(
        distance_matrix,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    d = np.maximum(d, 0.0)
    d = np.minimum(d, d.T)
    np.fill_diagonal(d, np.inf)

    finite_edges = d[np.isfinite(d)]
    if finite_edges.size == 0:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    finite_edges = finite_edges[finite_edges > 0.0]
    global_scale = float(np.median(finite_edges)) if finite_edges.size else 1.0
    global_scale = max(global_scale, eps)

    order = np.argsort(d, axis=1, kind="mergesort")
    sorted_d = np.take_along_axis(d, order, axis=1)

    nearest = sorted_d[:, 0]
    finite_nearest = nearest[np.isfinite(nearest)]
    robust_scale = (
        float(np.median(finite_nearest))
        if finite_nearest.size
        else global_scale
    )
    robust_scale = max(robust_scale, eps)

    usable = min(4, n - 1)
    neighborhood = sorted_d[:, :usable]
    finite_mask = np.isfinite(neighborhood)
    local_scale = np.divide(
        np.where(finite_mask, neighborhood, 0.0).sum(axis=1),
        finite_mask.sum(axis=1),
        out=np.full(n, robust_scale, dtype=float),
        where=finite_mask.sum(axis=1) > 0,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(np.outer(local_scale, local_scale))
    pair_scale = np.maximum(pair_scale, eps)

    safe_d = np.where(np.isfinite(d), d, robust_scale * 10.0)
    normalized_distance = safe_d / (pair_scale + robust_scale)

    distance_score = 1.0 / (1.0 + normalized_distance**2)

    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = 1.0 / np.power(1.0 + ranks, 0.85)
    reciprocal_rank = np.sqrt(rank_score * rank_score.T)

    nearest_index = order[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (nearest_index[:, None] == indices[None, :])
        & (nearest_index[None, :] == indices[:, None])
    ).astype(float)

    local_compatibility = np.exp(
        -safe_d / (np.sqrt(np.outer(local_scale, local_scale)) + robust_scale)
    )

    scores = (
        distance_score
        * (0.55 + 0.30 * reciprocal_rank + 0.15 * local_compatibility)
        * (1.0 + 0.40 * mutual_nearest)
    )

    scores = 0.5 * (scores + scores.T)
    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores, initial=0.0))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float)
