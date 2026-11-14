import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    symmetric_valid = valid | valid.T
    both_valid = valid & valid.T

    symmetric_d = np.where(
        both_valid,
        0.5 * (d + d.T),
        np.where(valid, d, d.T),
    )
    symmetric_d = np.where(symmetric_valid, symmetric_d, np.inf)
    np.fill_diagonal(symmetric_d, np.inf)

    positive_d = np.where(symmetric_valid & (symmetric_d > 0), symmetric_d, np.inf)
    positive_values = positive_d[np.isfinite(positive_d)]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    positive_order = np.sort(positive_d, axis=1)
    positive_count = np.sum(np.isfinite(positive_order), axis=1)
    local_k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    local_samples = positive_order[:, :local_k]
    local_scale = np.nanmedian(
        np.where(np.isfinite(local_samples), local_samples, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )

    safe_local_scale = np.maximum(local_scale, eps)
    denom = np.sqrt(
        safe_local_scale[:, None] * safe_local_scale[None, :]
    )

    finite_d = np.where(symmetric_valid, symmetric_d, np.inf)

    global_ratio = finite_d / global_scale
    local_ratio = finite_d / denom

    global_score = np.exp(-np.minimum(global_ratio, 40.0))
    local_score = np.exp(-np.minimum(local_ratio, 40.0))

    order = np.argsort(finite_d, axis=1, kind="stable")
    ranks = np.empty_like(order)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n)[None, :]

    rank_score = np.exp(-0.58 * ranks)
    rank_score = np.where(symmetric_valid, rank_score, 0.0)

    nearest = (ranks == 0) & symmetric_valid
    second = (ranks == 1) & symmetric_valid
    third = (ranks == 2) & symmetric_valid

    mutual_nearest = nearest & nearest.T

    nearest_support = nearest.astype(float)
    nearest_support += 0.5 * mutual_nearest.astype(float)

    third_bonus = 0.12 * third.astype(float)

    score = (
        0.30 * global_score
        + 0.34 * local_score
        + 0.22 * rank_score
        + 0.08 * nearest_support
        + 0.04 * second.astype(float)
        + third_bonus
    )

    adaptive_neighborhood = max(3, int(np.ceil(np.sqrt(n))))
    useful = ranks < min(n - 1, adaptive_neighborhood)
    score = np.where(useful, score, 0.86 * score)

    score = np.where(symmetric_valid, score, 0.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
