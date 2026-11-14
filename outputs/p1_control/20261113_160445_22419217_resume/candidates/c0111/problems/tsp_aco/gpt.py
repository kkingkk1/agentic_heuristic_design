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

    safe_d = np.where(valid, d, np.inf)
    order = np.argsort(safe_d, axis=1, kind="stable")
    sorted_d = np.take_along_axis(safe_d, order, axis=1)

    valid_counts = np.sum(valid, axis=1)
    positive = d[valid & (d > 0)]

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = 1.0

    global_scale = max(global_scale, np.finfo(float).eps)

    k = min(3, n - 1)
    local_values = sorted_d[:, :k]
    local_scales = np.where(
        np.isfinite(local_values),
        local_values,
        global_scale,
    )

    local_scales = np.median(local_scales, axis=1)
    local_scales = np.where(
        np.isfinite(local_scales) & (local_scales > 0),
        local_scales,
        global_scale,
    )

    pair_scale = np.sqrt(
        local_scales[:, None] * local_scales[None, :]
    )

    normalized_cost = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=pair_scale > 0,
    )

    global_score = np.exp(
        -np.minimum(d / global_scale, 40.0)
    )
    local_score = 1.0 / (1.0 + np.minimum(normalized_cost, 1e12))
    reciprocal_score = 1.0 / (
        1.0 + np.minimum(normalized_cost, 1e12) ** 0.75
    )

    ranks = np.empty_like(order)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n)[None, :]

    rank_score = np.exp(-0.55 * ranks.astype(float))
    rank_score = np.where(valid, rank_score, 0.0)

    nearest = (ranks == 0) & valid
    second = (ranks == 1) & valid
    third = (ranks == 2) & valid

    mutual_nearest = nearest & nearest.T & valid & valid.T
    reciprocal_support = (
        nearest & nearest.T
    ).astype(float) + 0.5 * (
        second & second.T
    ).astype(float)

    score = (
        0.24 * global_score
        + 0.30 * local_score
        + 0.20 * reciprocal_score
        + 0.14 * rank_score
        + 0.08 * nearest.astype(float)
        + 0.035 * second.astype(float)
        + 0.025 * third.astype(float)
        + 0.18 * mutual_nearest.astype(float)
        + 0.06 * reciprocal_support
    )

    neighborhood_limit = np.minimum(
        np.maximum(valid_counts - 1, 0),
        3,
    )
    useful = ranks <= neighborhood_limit[:, None]
    score = np.where(useful, score, 0.86 * score)

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
