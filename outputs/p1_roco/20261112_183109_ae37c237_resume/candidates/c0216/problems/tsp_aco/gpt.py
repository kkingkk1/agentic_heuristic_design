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

    symmetric_distance = 0.5 * (distances + distances.T)
    symmetric_distance[~valid] = np.inf
    np.fill_diagonal(symmetric_distance, np.inf)

    finite_values = symmetric_distance[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = float(np.max(finite_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(finite_values))
    penalty = max(finite_max, global_scale) * 4.0 + global_scale

    work = np.where(valid, symmetric_distance, penalty)
    np.fill_diagonal(work, penalty)

    nearest_two = np.partition(work, kth=1, axis=1)[:, :2]
    best_distance = nearest_two[:, 0]
    second_distance = nearest_two[:, 1]

    local_scale = 0.5 * (best_distance + second_distance)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    regret = np.divide(
        second_distance - best_distance,
        np.maximum(second_distance, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second_distance),
    )
    regret = np.clip(regret, 0.0, 1.0)

    nearest_neighbor = np.argmin(work, axis=1)
    indices = np.arange(n)
    mutual_nearest = (
        (nearest_neighbor[:, None] == indices[None, :])
        & (nearest_neighbor[None, :] == indices[:, None])
    )

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
        -np.clip(1.05 * relative_distance, 0.0, 60.0)
    )

    endpoint_quality = np.exp(
        -np.clip(0.5 * (row_ratio + col_ratio), 0.0, 60.0)
    )

    soft_rank_quality = np.exp(
        -np.clip(
            np.sqrt(
                np.maximum(row_ratio * col_ratio, 0.0)
            ),
            0.0,
            60.0,
        )
    )

    competition = np.sum(
        valid & (
            symmetric_distance
            <= second_distance[:, None]
        ),
        axis=1,
        dtype=float,
    )
    scarcity = 1.0 + 0.10 / np.sqrt(
        np.maximum(competition[:, None], 1.0)
        * np.maximum(competition[None, :], 1.0)
    )

    regret_bonus = 1.0 + 0.28 * (
        regret[:, None] + regret[None, :]
    )
    mutual_bonus = np.where(mutual_nearest, 1.24, 1.0)
    long_edge_penalty = np.exp(
        -0.58 * np.maximum(relative_distance - 1.0, 0.0)
    )

    scores = (
        distance_quality
        * (0.50 + 0.28 * endpoint_quality + 0.22 * soft_rank_quality)
        * regret_bonus
        * mutual_bonus
        * scarcity
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
