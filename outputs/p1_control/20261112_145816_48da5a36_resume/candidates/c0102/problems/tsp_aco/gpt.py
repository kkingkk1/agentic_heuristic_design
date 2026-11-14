import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0.0)
    valid = finite_positive & finite_positive.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = np.zeros_like(d, dtype=float)
    symmetric_d[valid] = 0.5 * (d[valid] + d.T[valid])

    positive_values = symmetric_d[valid]
    eps = np.finfo(float).eps
    fallback = max(float(np.median(positive_values)), eps)
    replacement = max(float(np.max(positive_values)) * 10.0, fallback)

    work = np.where(valid, symmetric_d, replacement)
    np.fill_diagonal(work, replacement)

    neighbor_values = np.sort(work, axis=1)
    kth = min(2, n - 2)
    local_scale = np.maximum(neighbor_values[:, kth], eps)

    normalized_length = work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    global_scale = max(float(np.median(positive_values)), eps)
    global_length = work / global_scale

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    ranks[~valid] = float(n)

    mutual_rank_log = -0.5 * (
        np.log1p(ranks) + np.log1p(ranks.T)
    )

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / (
        local_scale[:, None]
        + local_scale[None, :]
        + eps
    )

    log_score = (
        -1.05 * normalized_length
        -0.30 * global_length
        +1.55 * mutual_rank_log
        -0.75 * scale_difference
    )

    finite_scores = valid & np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - maximum, -745.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
