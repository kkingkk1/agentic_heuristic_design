import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, normalized edge-promise matrix for stochastic TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    positive = valid & (d > 0.0)
    positive_values = d[positive]

    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    fallback = (
        float(np.max(positive_values))
        if positive_values.size
        else global_scale
    )
    replacement = max(fallback * 10.0, global_scale, eps)

    work = np.where(valid, np.maximum(d, 0.0), replacement)
    np.fill_diagonal(work, replacement)

    # Smooth local scale from the two nearest finite-safe neighbors.
    neighbor_count = min(2, n - 1)
    local_scale = np.partition(
        work,
        kth=neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count].mean(axis=1)
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = work / endpoint_scale
    relative_length = np.minimum(relative_length, 1e12)

    # Stable row-wise ranks reward edges preferred by both endpoints.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    normalized_ranks = ranks / max(n - 1, 1)

    reciprocal_rank = 1.0 / (
        1.0 + normalized_ranks + normalized_ranks.T
    )

    # Scale compatibility penalizes edges joining very different neighborhoods.
    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = (
        local_scale[:, None] + local_scale[None, :] + eps
    )
    compatibility_penalty = scale_difference / scale_sum

    absolute_length = work / global_scale

    log_score = (
        -0.80 * relative_length
        -0.25 * absolute_length
        + 1.85 * np.log(np.maximum(reciprocal_rank, eps))
        -0.75 * compatibility_penalty
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.exp(np.clip(log_score - maximum, -745.0, 0.0))
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
