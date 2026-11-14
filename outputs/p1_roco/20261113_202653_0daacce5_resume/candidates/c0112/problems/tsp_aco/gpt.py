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
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d)
    positive = d[valid & (d > 0.0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    global_scale = float(np.median(positive))
    global_scale = max(global_scale, np.finfo(float).eps)

    safe_d = np.where(valid, d, np.inf)
    eps = np.finfo(float).eps * max(global_scale, 1.0)

    local_scales = np.empty(n, dtype=float)
    for i in range(n):
        values = safe_d[i, valid[i]]
        if values.size:
            k = min(4, values.size)
            local_scales[i] = float(np.mean(np.partition(values, k - 1)[:k]))
        else:
            local_scales[i] = global_scale

    local_scales = np.maximum(local_scales, 0.05 * global_scale)
    pair_scale = np.sqrt(
        local_scales[:, None] * local_scales[None, :]
    )

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (safe_d[valid] + eps)

    row_max = np.maximum(
        np.max(inverse, axis=1, keepdims=True),
        eps,
    )
    row_strength = inverse / row_max
    mutual_strength = np.sqrt(row_strength * row_strength.T)

    normalized_distance = np.zeros((n, n), dtype=float)
    normalized_distance[valid] = (
        safe_d[valid] / (pair_scale[valid] + eps)
    )

    local_attraction = np.zeros((n, n), dtype=float)
    local_attraction[valid] = np.exp(
        -np.minimum(normalized_distance[valid], 40.0)
    )

    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.full((n, n), n, dtype=float)
    rank_values = np.arange(n, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = rank_values

    rank_score = np.zeros((n, n), dtype=float)
    rank_score[valid] = 1.0 / (1.0 + ranks[valid])
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    scale_match = np.zeros((n, n), dtype=float)
    scale_match[valid] = np.exp(
        -np.abs(
            np.log((local_scales[:, None] + eps)
                   / (local_scales[None, :] + eps))
        )[valid]
    )

    score = (
        0.40 * mutual_strength
        + 0.30 * local_attraction
        + 0.20 * mutual_rank
        + 0.10 * scale_match
    )

    score[valid] *= np.exp(
        -0.15 * np.minimum(normalized_distance[valid], 20.0)
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
