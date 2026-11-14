import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return scale-invariant symmetric edge-priority indicators for TSP."""
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

    valid_values = symmetric_distance[valid]
    positive_values = valid_values[valid_values > 0.0]
    eps = np.finfo(float).eps

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

    neighbor_count = min(4, n - 1)
    nearest = np.partition(
        work,
        kth=neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]

    nearest_valid = nearest < penalty
    neighbor_totals = np.sum(
        np.where(nearest_valid, nearest, 0.0),
        axis=1,
    )
    neighbor_counts = np.sum(nearest_valid, axis=1)

    local_scale = np.divide(
        neighbor_totals,
        np.maximum(neighbor_counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=neighbor_counts > 0,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    best_distance = nearest[:, 0]
    second_distance = (
        nearest[:, 1]
        if neighbor_count >= 2
        else best_distance.copy()
    )

    safe_best = np.where(
        np.isfinite(best_distance),
        best_distance,
        local_scale,
    )
    safe_second = np.where(
        np.isfinite(second_distance) & (second_distance < penalty),
        second_distance,
        safe_best,
    )

    regret = np.divide(
        safe_second - safe_best,
        np.maximum(safe_second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(safe_second),
    )
    regret = np.clip(regret, 0.0, 1.0)

    nearest_neighbor = np.argmin(work, axis=1)
    indices = np.arange(n)
    mutual_nearest = (
        (nearest_neighbor[:, None] == indices[None, :])
        & (nearest_neighbor[None, :] == indices[:, None])
        & valid
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
        -np.clip(1.16 * relative_distance, 0.0, 60.0)
    )

    endpoint_quality = np.exp(
        -np.clip(0.38 * (row_ratio + col_ratio), 0.0, 60.0)
    )

    soft_rank_quality = np.exp(
        -np.clip(
            0.5 * np.sqrt(
                np.maximum(row_ratio * col_ratio, 0.0)
            ),
            0.0,
            60.0,
        )
    )

    competition = np.sum(
        valid & (
            symmetric_distance <= safe_second[:, None]
        ),
        axis=1,
        dtype=float,
    )

    scarcity = 1.0 + 0.08 / np.sqrt(
        np.maximum(competition[:, None], 1.0)
        * np.maximum(competition[None, :], 1.0)
    )

    urgency = 1.0 + 0.20 * (
        regret[:, None] + regret[None, :]
    )

    mutual_bonus = np.where(mutual_nearest, 1.20, 1.0)

    long_edge_penalty = np.exp(
        -0.48 * np.maximum(relative_distance - 1.0, 0.0)
    )

    scores = (
        distance_quality
        * (
            0.56
            + 0.24 * endpoint_quality
            + 0.20 * soft_rank_quality
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
