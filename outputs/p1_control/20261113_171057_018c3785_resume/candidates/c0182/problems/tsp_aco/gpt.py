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
    finite_nearest = nearest[np.isfinite(nearest)]

    if finite_nearest.size == 0:
        return np.zeros_like(d, dtype=float)

    positive_nearest = finite_nearest[finite_nearest > 0]
    if positive_nearest.size:
        global_scale = float(np.median(positive_nearest))
    else:
        positive_edges = d[valid & (d > 0)]
        global_scale = (
            float(np.median(positive_edges))
            if positive_edges.size
            else 1.0
        )

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    local_scale = np.where(
        np.isfinite(nearest) & (nearest > 0),
        nearest,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    normalized_distance = np.clip(d / global_scale, 0.0, 700.0)
    short_edge = np.exp(-normalized_distance)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = np.clip(d / pair_scale, 0.0, 700.0)
    relative_edge = np.exp(-relative_distance)

    ranks = np.argsort(np.argsort(safe_d, axis=1), axis=1)
    mutual_best = (
        (ranks == 0) & (ranks.T == 0)
    ).astype(float)

    score = (
        0.68 * short_edge
        + 0.28 * relative_edge
        + 0.04 * mutual_best
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
