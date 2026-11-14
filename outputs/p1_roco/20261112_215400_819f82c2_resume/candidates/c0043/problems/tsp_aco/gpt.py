import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    positive = d[valid & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    safe = np.where(valid, d, np.inf)
    finite_rows = np.where(np.isfinite(safe), safe, np.nan)

    nearest = np.min(safe, axis=1)
    row_median = np.nanmedian(finite_rows, axis=1)

    nearest[~np.isfinite(nearest)] = global_scale
    row_median[~np.isfinite(row_median)] = global_scale

    local_scale = 0.5 * (nearest + row_median)
    local_scale[~np.isfinite(local_scale) | (local_scale <= 0.0)] = global_scale
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    # Endpoint-specific distance preferences.
    row_preference = np.zeros((n, n), dtype=float)
    valid_values = np.isfinite(safe)
    row_preference[valid_values] = np.exp(
        -np.clip(
            safe[valid_values]
            / local_scale[np.repeat(np.arange(n), n)[valid_values.ravel()]],
            0.0,
            50.0,
        )
    )

    # Rank preference from each endpoint.
    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_preference = 1.0 / (1.0 + ranks)
    rank_preference[~valid] = 0.0

    bidirectional_distance = np.sqrt(
        row_preference * row_preference.T
    )
    bidirectional_rank = np.sqrt(
        rank_preference * rank_preference.T
    )

    nearest_neighbor = np.argmin(safe, axis=1)
    mutual = np.zeros((n, n), dtype=float)

    for i, j in enumerate(nearest_neighbor):
        if np.isfinite(safe[i, j]) and nearest_neighbor[j] == i:
            mutual[i, j] = 1.0

    scores = (
        0.60 * bidirectional_distance
        + 0.30 * bidirectional_rank
        + 0.10 * mutual
    )

    scores = 0.5 * (scores + scores.T)
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
