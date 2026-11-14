import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return scale-invariant symmetric edge priorities for TSP."""
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

    valid_values = d[valid]
    positive_values = valid_values[valid_values > 0.0]

    eps = np.finfo(float).eps
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.max(valid_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(valid_values))
    penalty = max(finite_max, global_scale) * 8.0 + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(work, order, axis=1)

    degree = np.sum(valid, axis=1)
    best = sorted_distances[:, 0]

    if n > 1:
        second = sorted_distances[:, 1]
    else:
        second = best.copy()

    second = np.where(degree >= 2, second, best)

    neighbor_count = min(3, n - 1)
    nearest_block = sorted_distances[:, :neighbor_count]
    nearest_valid = np.arange(neighbor_count)[None, :] < np.minimum(
        degree[:, None], neighbor_count
    )
    local_sum = np.sum(
        np.where(nearest_valid, nearest_block, 0.0),
        axis=1,
    )
    local_denominator = np.maximum(
        np.sum(nearest_valid, axis=1),
        1,
    )
    local_scale = local_sum / local_denominator
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    normalized_rank = np.divide(
        ranks,
        np.maximum(degree[:, None] - 1.0, 1.0),
    )
    normalized_rank = np.clip(normalized_rank, 0.0, 1.0)

    rank_term = np.exp(
        -0.78 * (normalized_rank + normalized_rank.T)
    )

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        work,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=geometric_scale > 0.0,
    )

    distance_term = np.exp(
        -np.clip(1.12 * relative_distance, 0.0, 60.0)
    )

    endpoint_ratio = (
        np.divide(
            work,
            local_scale[:, None],
            out=np.full((n, n), np.inf, dtype=float),
            where=local_scale[:, None] > 0.0,
        )
        + np.divide(
            work,
            local_scale[None, :],
            out=np.full((n, n), np.inf, dtype=float),
            where=local_scale[None, :] > 0.0,
        )
    )

    endpoint_term = np.exp(
        -np.clip(0.34 * endpoint_ratio, 0.0, 60.0)
    )

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second) & (second > 0.0),
    )
    regret = np.clip(regret, 0.0, 1.0)

    regret_term = 1.0 + 0.38 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)
    has_neighbor = degree > 0

    mutual_nearest = (
        has_neighbor[:, None]
        & has_neighbor[None, :]
        & (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_term = np.where(mutual_nearest, 1.42, 1.0)

    long_edge_term = np.exp(
        -0.62 * np.maximum(relative_distance - 1.0, 0.0)
    )

    scores = (
        distance_term
        * (0.46 + 0.31 * rank_term + 0.23 * endpoint_term)
        * regret_term
        * mutual_term
        * long_edge_term
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
