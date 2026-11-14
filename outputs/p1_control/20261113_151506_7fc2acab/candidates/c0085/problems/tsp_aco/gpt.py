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
    nearest = np.min(safe_d, axis=1)

    finite_nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    positive_valid = d[valid & (d > 0)]

    if finite_nearest.size:
        global_scale = float(np.median(finite_nearest))
    elif positive_valid.size:
        global_scale = float(np.median(positive_valid))
    else:
        global_scale = 1.0

    global_scale = max(global_scale, np.finfo(float).eps)

    local_scale = np.where(
        np.isfinite(nearest) & (nearest > 0),
        nearest,
        global_scale,
    )

    local_ratio = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    local_score = np.exp(-np.minimum(local_ratio, 40.0))

    global_ratio = d / global_scale
    global_score = np.exp(-np.minimum(global_ratio, 40.0))

    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty_like(order)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n)[None, :]

    rank_score = np.exp(-0.72 * np.maximum(ranks, 0))
    rank_score = np.where(valid, rank_score, 0.0)

    nearest_bonus = ((ranks == 0) & valid).astype(float)
    second_bonus = ((ranks == 1) & valid).astype(float)
    mutual_bonus = (
        (ranks == 0) & (ranks.T == 0) & valid & valid.T
    ).astype(float)

    score = (
        0.34 * global_score
        + 0.34 * local_score
        + 0.18 * rank_score
        + 0.08 * nearest_bonus
        + 0.04 * second_bonus
        + 0.18 * mutual_bonus
    )

    useful = ranks <= 3
    score = np.where(useful, score, 0.82 * score)

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
