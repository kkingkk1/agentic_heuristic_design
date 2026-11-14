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
    nearest_index = np.argmin(safe_d, axis=1)
    nearest_distance = safe_d[
        np.arange(n), nearest_index
    ]

    has_neighbor = np.isfinite(nearest_distance)
    if not np.any(has_neighbor):
        return np.zeros_like(d, dtype=float)

    positive_nearest = nearest_distance[
        has_neighbor & (nearest_distance > 0)
    ]

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
        has_neighbor & (nearest_distance > 0),
        nearest_distance,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    clipped_distance = np.clip(d / global_scale, 0.0, 700.0)
    short_edge = np.exp(-clipped_distance)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=pair_scale > 0,
    )
    relative_edge = np.exp(-np.clip(relative_distance, 0.0, 700.0))

    mutual_best = (
        has_neighbor[:, None]
        & has_neighbor[None, :]
        & (nearest_index[:, None] == np.arange(n)[None, :])
        & (nearest_index[None, :] == np.arange(n)[:, None])
    ).astype(float)

    score = (
        0.62 * short_edge
        + 0.30 * relative_edge
        + 0.08 * mutual_best
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
