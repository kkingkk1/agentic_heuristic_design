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

    degree = valid.sum(axis=1)
    k = min(4, n - 1)

    order = np.argsort(d, axis=1, kind="stable")
    sorted_d = np.take_along_axis(d, order, axis=1)

    neighbor_mask = (
        np.arange(k)[None, :] < np.minimum(degree[:, None], k)
    )
    nearest_block = sorted_d[:, :k]

    local_sum = np.sum(
        np.where(neighbor_mask, nearest_block, 0.0),
        axis=1,
    )
    local_count = np.maximum(neighbor_mask.sum(axis=1), 1)

    local_scale = local_sum / local_count
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    best = np.where(degree > 0, sorted_d[:, 0], global_scale)

    second = np.where(
        degree >= 2,
        sorted_d[:, 1],
        2.0 * np.maximum(best, eps),
    )
    second = np.where(np.isfinite(second), second, 2.0 * best)

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second),
    )
    regret = np.clip(regret, 0.0, 1.0)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        d,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    distance_quality = np.exp(
        -np.clip(1.08 * relative_distance, 0.0, 60.0)
    )

    row_ratio = np.divide(
        d,
        local_scale[:, None],
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        local_scale[None, :],
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    endpoint_quality = np.exp(
        -np.clip(0.5 * (row_ratio + col_ratio), 0.0, 60.0)
    )

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = np.maximum(
        1.0,
        0.85 * np.sqrt(np.maximum(degree, 1.0)),
    )
    rank_quality = np.exp(
        -0.5
        * (
            ranks / rank_scale[:, None]
            + ranks / rank_scale[None, :]
        )
    )

    nearest = order[:, 0]
    indices = np.arange(n)

    mutual_nearest = (
        (degree[:, None] > 0)
        & (degree[None, :] > 0)
        & (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )

    mutual_bonus = np.where(mutual_nearest, 1.22, 1.0)

    regret_bonus = 1.0 + 0.26 * (
        regret[:, None] + regret[None, :]
    )

    threshold = np.where(degree >= 2, second, best)
    competition = np.sum(
        valid & (d <= threshold[:, None]),
        axis=1,
        dtype=float,
    )

    scarcity = 1.0 + 0.08 / np.sqrt(
        np.maximum(competition[:, None], 1.0)
        * np.maximum(competition[None, :], 1.0)
    )

    proximity = np.exp(
        -np.clip(relative_distance, 0.0, 40.0)
    )
    proximity[~valid] = 0.0
    np.fill_diagonal(proximity, 0.0)

    profile_norm = np.sqrt(
        np.sum(proximity * proximity, axis=1)
    )

    overlap = np.divide(
        proximity @ proximity.T,
        profile_norm[:, None] * profile_norm[None, :],
        out=np.zeros((n, n), dtype=float),
        where=(
            profile_norm[:, None] * profile_norm[None, :]
        ) > 0.0,
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge_bonus = 1.0 + 0.06 * np.clip(
        (1.0 - overlap)
        * (1.0 - np.exp(-np.maximum(relative_distance, 0.0))),
        0.0,
        1.0,
    )

    long_edge_penalty = np.exp(
        -0.48 * np.maximum(relative_distance - 1.0, 0.0)
    )

    scores = (
        distance_quality
        * (0.48 + 0.30 * endpoint_quality + 0.22 * rank_quality)
        * regret_bonus
        * mutual_bonus
        * scarcity
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
