import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, rank-aware edge priors for an undirected TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    eps = np.finfo(float).eps
    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    safe = np.where(valid, d, np.inf)
    ordered = np.sort(safe, axis=1)

    k = min(3, n - 1)
    neighborhood = ordered[:, :k]
    positive = neighborhood[np.isfinite(neighborhood) & (neighborhood > 0.0)]

    if positive.size == 0:
        scale = 1.0
    else:
        scale = max(float(np.median(positive)), eps)

    local_scale = np.median(
        np.where(np.isfinite(neighborhood), neighborhood, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        scale,
    )
    local_scale = np.maximum(local_scale, eps)

    positive_all = d[valid & (d > 0.0)]
    global_scale = (
        max(float(np.median(positive_all)), eps)
        if positive_all.size
        else scale
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    local_scale_sum = local_scale[:, None] + local_scale[None, :]

    global_signal = np.exp(-np.clip(d / global_scale, 0.0, 60.0))
    local_signal = np.exp(-np.clip(d / pair_scale, 0.0, 60.0))

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks[~valid] = np.inf

    rank_signal = np.exp(-0.7 * np.minimum(ranks, n))
    rank_signal[~valid] = 0.0

    mutual_nearest = (
        (ranks == 0.0) & (ranks.T == 0.0) & valid & valid.T
    ).astype(float)

    savings = np.maximum(local_scale_sum - d, 0.0)
    savings /= np.maximum(local_scale_sum, eps)

    long_edge = d > 1.8 * local_scale_sum
    suppression = np.where(long_edge, 0.4, 1.0)

    score = (
        0.34 * global_signal
        + 0.34 * local_signal
        + 0.16 * rank_signal
        + 0.10 * mutual_nearest
        + 0.06 * savings
    )

    score *= suppression
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
