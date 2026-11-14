import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, scale-invariant edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    valid_values = d[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.max(valid_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(valid_values))
    penalty = max(finite_max, global_scale) * 5.0 + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    nearest = np.argmin(work, axis=1)
    nearest_distance = np.min(work, axis=1)

    if n > 2:
        nearest_two = np.partition(work, kth=1, axis=1)[:, :2]
        best = nearest_two[:, 0]
        second = nearest_two[:, 1]
    else:
        best = nearest_distance
        second = nearest_distance

    local_scale = 0.5 * (best + second)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        d,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (geometric_scale > 0.0),
    )

    row_ratio = np.divide(
        d,
        local_scale[:, None],
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )
    column_ratio = np.divide(
        d,
        local_scale[None, :],
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    distance_term = np.exp(
        -np.clip(1.05 * relative_distance, 0.0, 60.0)
    )

    endpoint_quality = np.exp(
        -np.clip(
            0.5 * (row_ratio + column_ratio),
            0.0,
            60.0,
        )
    )

    rank_proxy = np.exp(
        -np.clip(
            np.sqrt(np.maximum(row_ratio * column_ratio, 0.0)),
            0.0,
            60.0,
        )
    )

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second),
    )
    regret = np.clip(regret, 0.0, 1.0)

    regret_bonus = 1.0 + 0.34 * (
        regret[:, None] + regret[None, :]
    )

    mutual_nearest = (
        (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.32, 1.0)

    long_edge_penalty = np.exp(
        -0.58 * np.maximum(relative_distance - 1.0, 0.0)
    )

    scores = (
        distance_term
        * (
            0.50
            + 0.30 * endpoint_quality
            + 0.20 * rank_proxy
        )
        * regret_bonus
        * mutual_bonus
        * long_edge_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
