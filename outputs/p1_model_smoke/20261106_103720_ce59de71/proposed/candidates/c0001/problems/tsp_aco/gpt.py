import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    off_diag = ~np.eye(n, dtype=bool)
    positive = d[off_diag]
    positive = positive[positive > 0]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = np.median(positive)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.mean(positive)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    x = d / scale
    eps = np.finfo(float).eps

    # Several distance scales capture both local and moderately global structure.
    scales = np.array([0.35, 0.8, 1.8], dtype=float)
    affinity = np.zeros((n, n), dtype=float)

    for temperature in scales:
        affinity += np.exp(-x / temperature)

    affinity /= float(scales.size)
    np.fill_diagonal(affinity, 0.0)

    # Endpoint competition: an edge is less valuable if either endpoint
    # has many similarly attractive alternatives.
    row_peak = np.max(affinity, axis=1, keepdims=True)
    relative = affinity / (row_peak + eps)
    competition = 1.0 / (
        1.0
        + np.sum(relative, axis=1, keepdims=True)
        + np.sum(relative, axis=0, keepdims=True)
    )

    # Triangle closure: reward edges whose endpoints have compatible
    # low-cost connections through a third node.
    closure = np.zeros((n, n), dtype=float)
    for k in range(n):
        via_k = x[:, k:k + 1] + x[k:k + 1, :]
        closure += np.exp(-via_k)
    closure /= max(n - 2, 1)
    np.fill_diagonal(closure, 0.0)

    # Robust global rank term avoids dependence on the absolute distance scale.
    sorted_values = np.sort(positive)
    ranks = np.searchsorted(sorted_values, d, side="right") / sorted_values.size
    rank_score = 1.0 - ranks
    rank_score[~off_diag] = 0.0

    score = (
        0.48 * affinity
        + 0.27 * closure
        + 0.17 * rank_score
        + 0.08 * competition * (affinity + eps)
    )

    score = 0.5 * (score + score.T)
    score[~off_diag] = np.maximum(score[~off_diag], 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
