import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric normalized edge-priority indicators for TSP."""
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

    symmetric_distance = 0.5 * (distances + distances.T)
    symmetric_distance[~valid] = np.inf
    np.fill_diagonal(symmetric_distance, np.inf)

    eps = np.finfo(float).eps
    valid_values = symmetric_distance[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif valid_values.size:
        global_scale = float(np.max(valid_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    maximum_distance = float(np.max(valid_values))
    penalty = max(maximum_distance, global_scale) * 8.0 + global_scale

    work = np.where(valid, symmetric_distance, penalty)
    np.fill_diagonal(work, penalty)

    k = min(4, n - 1)
    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    nearest_valid = nearest < penalty

    neighbor_sum = np.sum(
        np.where(nearest_valid, nearest, 0.0),
        axis=1,
    )
    neighbor_count = np.sum(nearest_valid, axis=1)

    local_scale = np.divide(
        neighbor_sum,
        np.maximum(neighbor_count, 1),
        out=np.full(n, global_scale, dtype=float),
        where=neighbor_count > 0,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    best = nearest[:, 0]
    second = nearest[:, 1] if k >= 2 else best.copy()

    safe_best = np.where(
        np.isfinite(best) & (best < penalty),
        best,
        local_scale,
    )
    safe_second = np.where(
        np.isfinite(second) & (second < penalty),
        second,
        safe_best,
    )

    regret = np.divide(
        safe_second - safe_best,
        np.maximum(safe_second, eps),
    )
    regret = np.clip(regret, 0.0, 1.0)

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    relative_distance = np.divide(
        symmetric_distance,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (geometric_scale > 0.0),
    )

    row_ratio = np.divide(
        symmetric_distance,
        row_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (row_scale > 0.0),
    )
    col_ratio = np.divide(
        symmetric_distance,
        col_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (col_scale > 0.0),
    )

    distance_quality = np.exp(
        -np.clip(1.10 * relative_distance, 0.0, 60.0)
    )

    endpoint_quality = np.exp(
        -np.clip(0.32 * (row_ratio + col_ratio), 0.0, 60.0)
    )

    soft_rank_quality = np.exp(
        -np.clip(
            0.18 * (row_ratio + col_ratio),
            0.0,
            60.0,
        )
    )

    competition = np.sum(
        valid & (symmetric_distance <= safe_second[:, None]),
        axis=1,
        dtype=float,
    )

    scarcity = 1.0 + 0.08 / np.sqrt(
        np.maximum(competition[:, None], 1.0)
        * np.maximum(competition[None, :], 1.0)
    )

    urgency = 1.0 + 0.14 * (
        regret[:, None] + regret[None, :]
    )

    nearest_neighbor = np.argmin(work, axis=1)
    indices = np.arange(n)
    mutual_nearest = (
        (nearest_neighbor[:, None] == indices[None, :])
        & (nearest_neighbor[None, :] == indices[:, None])
        & valid
    )
    mutual_bonus = np.where(mutual_nearest, 1.12, 1.0)

    long_edge_penalty = np.exp(
        -0.24 * np.maximum(relative_distance - 1.0, 0.0)
    )

    scores = (
        distance_quality
        * (
            0.62
            + 0.20 * endpoint_quality
            + 0.18 * soft_rank_quality
        )
        * scarcity
        * urgency
        * mutual_bonus
        * long_edge_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
