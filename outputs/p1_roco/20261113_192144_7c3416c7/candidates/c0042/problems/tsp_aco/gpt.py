import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-selection priors for TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(
        d,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)

    valid = np.isfinite(d)
    np.fill_diagonal(valid, False)
    np.fill_diagonal(d, np.inf)

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)
    eps = np.finfo(float).eps * max(scale, 1.0)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe_d = np.where(valid, d, np.inf)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (safe_d[valid] + eps)

    row_max = np.max(inverse, axis=1, keepdims=True)
    row_max = np.maximum(row_max, eps)
    row_preference = inverse / row_max

    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.full((n, n), n, dtype=float)
    rank_values = np.arange(n, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = rank_values
    rank_score = np.where(valid, 1.0 / (1.0 + ranks), 0.0)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    neighborhood_size = min(3, n - 1)
    local_scales = np.empty(n, dtype=float)

    for i in range(n):
        row_values = safe_d[i, valid[i]]
        if row_values.size == 0:
            local_scales[i] = scale
        else:
            local_scales[i] = float(
                np.mean(np.sort(row_values)[:neighborhood_size])
            )

    local_scales = np.maximum(local_scales, scale * 0.05)
    pair_scale = np.sqrt(
        local_scales[:, None] * local_scales[None, :]
    )

    contrast = np.zeros((n, n), dtype=float)
    contrast[valid] = np.exp(
        -safe_d[valid] / (pair_scale[valid] + eps)
    )

    score = (
        0.55 * row_preference
        + 0.25 * mutual_rank
        + 0.20 * contrast
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
