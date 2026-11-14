import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-priority matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite_positive = np.isfinite(d) & (d > 0.0)
    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    finite_positive = np.isfinite(d) & (d > 0.0)

    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[finite_positive]
    global_scale = max(float(np.median(positive_values)), eps)
    replacement = max(float(np.max(positive_values)) * 10.0, global_scale)

    work = np.where(finite_positive, d, replacement)
    np.fill_diagonal(work, replacement)

    # Use the second-nearest available edge when possible.
    kth = min(1, n - 2)
    local_scale = np.partition(work, kth=kth, axis=1)[:, kth]
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = np.maximum(work / endpoint_scale, 0.0)

    # Stable bilateral endpoint ranks.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    ranks /= max(n - 1, 1)

    bilateral_rank = np.sqrt(
        np.maximum(1.0 - ranks, eps)
        * np.maximum(1.0 - ranks.T, eps)
    )

    # Local-scale compatibility in log space.
    scale_gap = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )

    absolute_length = work / global_scale

    log_score = (
        -0.90 * np.minimum(relative_length, 1e6)
        -0.30 * np.minimum(absolute_length, 1e6)
        +0.95 * np.log(np.maximum(bilateral_rank, eps))
        -0.35 * np.minimum(scale_gap, 50.0)
    )

    valid_scores = log_score[valid]
    if valid_scores.size == 0 or not np.any(np.isfinite(valid_scores)):
        return np.zeros((n, n), dtype=float)

    log_score -= float(np.max(valid_scores))
    score = np.exp(np.clip(log_score, -745.0, 0.0))

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
