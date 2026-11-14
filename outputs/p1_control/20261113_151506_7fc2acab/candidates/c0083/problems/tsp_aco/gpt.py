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
    sorted_d = np.sort(safe_d, axis=1)

    nearest = sorted_d[:, 0]
    k = min(3, n - 1)
    local_neighbors = sorted_d[:, :k]
    finite_values = local_neighbors[np.isfinite(local_neighbors)]

    if finite_values.size == 0:
        return np.zeros_like(d, dtype=float)

    positive = finite_values[finite_values > 0]
    if positive.size:
        fallback = float(np.median(positive))
    else:
        fallback = 1.0

    nearest_positive = nearest[np.isfinite(nearest) & (nearest > 0)]
    if nearest_positive.size:
        global_scale = float(np.median(nearest_positive))
    else:
        global_scale = fallback

    global_scale = max(global_scale, np.finfo(float).eps)

    local_scale = np.median(
        np.where(np.isfinite(local_neighbors), local_neighbors, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    additive_scale = local_scale[:, None] + local_scale[None, :]

    global_shortness = np.exp(-d / global_scale)
    local_shortness = np.exp(-d / pair_scale)

    ranks = np.argsort(np.argsort(safe_d, axis=1), axis=1)
    rank_bonus = np.zeros_like(d, dtype=float)
    rank_bonus += (ranks <= 0).astype(float) * 1.00
    rank_bonus += (ranks == 1).astype(float) * 0.45
    rank_bonus += (ranks == 2).astype(float) * 0.20

    mutual_nearest = (
        (ranks == 0) & (ranks.T == 0)
    ).astype(float)

    savings = np.maximum(additive_scale - d, 0.0) / additive_scale

    implausibly_long = d > 1.8 * additive_scale
    suppression = np.where(implausibly_long, 0.35, 1.0)

    score = (
        0.30 * global_shortness
        + 0.30 * local_shortness
        + 0.18 * rank_bonus
        + 0.14 * mutual_nearest
        + 0.08 * savings
    )

    score *= suppression
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
