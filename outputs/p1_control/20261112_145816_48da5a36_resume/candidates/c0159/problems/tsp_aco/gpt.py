import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    valid = finite & finite.T & (d > 0.0) & (d.T > 0.0)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    symmetric_d = np.zeros_like(d, dtype=float)
    symmetric_d[valid] = 0.5 * (d[valid] + d.T[valid])

    positive_values = symmetric_d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    largest = float(np.max(positive_values))
    replacement = min(largest * 10.0, np.finfo(float).max)
    replacement = max(replacement, global_scale, eps)

    work = np.where(valid, symmetric_d, replacement)
    np.fill_diagonal(work, replacement)

    kth = min(2, n - 2)
    local_scale = np.partition(work, kth, axis=1)[:, kth]
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = work / endpoint_scale
    global_length = work / global_scale

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    reciprocal_rank = 1.0 / (1.0 + ranks + ranks.T)

    scale_difference = (
        np.abs(local_scale[:, None] - local_scale[None, :])
        / (local_scale[:, None] + local_scale[None, :] + eps)
    )

    log_score = (
        -1.10 * relative_length
        -0.28 * global_length
        +1.65 * np.log(reciprocal_rank)
        -0.85 * scale_difference
    )

    finite_scores = valid & np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    shift = float(np.max(log_score[finite_scores]))
    score = np.zeros_like(work, dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - shift, -745.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
