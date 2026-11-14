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
    elif valid_values.size:
        global_scale = float(np.max(valid_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    maximum_distance = float(np.max(valid_values))
    penalty = max(maximum_distance, global_scale) * 8.0 + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_work = np.take_along_axis(work, order, axis=1)

    degree = np.sum(valid, axis=1)
    k = min(4, n - 1)

    neighbor_values = sorted_work[:, :k]
    neighbor_valid = (
        np.arange(k, dtype=int)[None, :]
        < np.minimum(degree[:, None], k)
    )

    neighbor_sum = np.sum(
        np.where(neighbor_valid, neighbor_values, 0.0),
        axis=1,
    )
    neighbor_count = np.sum(neighbor_valid, axis=1)

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

    best = sorted_work[:, 0]
    second = sorted_work[:, 1] if n > 2 else best.copy()

    best = np.where(np.isfinite(best) & (best < penalty), best, local_scale)
    second = np.where(
        np.isfinite(second) & (second < penalty),
        second,
        best,
    )
    second = np.where(degree >= 2, second, best)

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second),
    )
    regret = np.clip(regret, 0.0, 1.0)

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    relative_distance = np.divide(
        d,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (geometric_scale > 0.0),
    )

    distance_quality = np.exp(
        -np.clip(1.15 * relative_distance, 0.0, 60.0)
    )

    row_ratio = np.divide(
        d,
        row_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (row_scale > 0.0),
    )
    col_ratio = np.divide(
        d,
        col_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (col_scale > 0.0),
    )

    endpoint_quality = np.exp(
        -np.clip(0.42 * (row_ratio + col_ratio), 0.0, 60.0)
    )

    rank_positions = np.empty((n, n), dtype=float)
    rank_positions[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)[None, :]
    rank_positions = rank_positions + 1.0

    rank_quality = np.exp(
        -np.clip(
            (rank_positions + rank_positions.T)
            / max(2.0, 1.35 * np.sqrt(float(n))),
            0.0,
            60.0,
        )
    )

    row_threshold = second[:, None]
    col_threshold = second[None, :]

    row_competition = np.sum(
        valid & (d <= row_threshold),
        axis=1,
        dtype=float,
    )
    col_competition = np.sum(
        valid & (d <= col_threshold),
        axis=0,
        dtype=float,
    )

    scarcity = 1.0 + 0.07 / np.sqrt(
        np.maximum(row_competition[:, None], 1.0)
        * np.maximum(col_competition[None, :], 1.0)
    )

    urgency = 1.0 + 0.16 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (degree[:, None] > 0)
        & (degree[None, :] > 0)
        & (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
        & valid
    )
    mutual_bonus = np.where(mutual_nearest, 1.18, 1.0)

    proximity = np.exp(
        -np.clip(relative_distance, 0.0, 40.0)
    )
    proximity[~valid] = 0.0
    np.fill_diagonal(proximity, 0.0)

    profile_norm = np.sqrt(np.sum(proximity * proximity, axis=1))
    denominator = profile_norm[:, None] * profile_norm[None, :]
    overlap = np.divide(
        proximity @ proximity.T,
        denominator,
        out=np.zeros((n, n), dtype=float),
        where=denominator > 0.0,
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge_bonus = np.clip(
        1.0
        + 0.06 * (1.0 - overlap)
        - 0.02 * overlap * np.exp(
            -np.clip(relative_distance, 0.0, 40.0)
        ),
        0.95,
        1.06,
    )

    long_edge_penalty = np.exp(
        -0.45 * np.maximum(relative_distance - 1.0, 0.0)
    )

    core_quality = (
        0.58 * distance_quality
        + 0.24 * endpoint_quality
        + 0.18 * rank_quality
    )

    scores = (
        core_quality
        * scarcity
        * urgency
        * mutual_bonus
        * bridge_bonus
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
