import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return scale-aware edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.max(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(finite_values))
    penalty = max(finite_max, global_scale) * 4.0 + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    k = min(4, n - 1)
    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    nearest_finite = np.isfinite(nearest)

    safe_nearest = np.where(nearest_finite, nearest, global_scale)
    neighbor_count = nearest_finite.sum(axis=1)
    local_scale = np.divide(
        safe_nearest.sum(axis=1),
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

    ordered = np.sort(work, axis=1)
    best = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, global_scale)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)

    regret = np.divide(
        safe_second - safe_best,
        np.maximum(safe_second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(safe_second),
    )
    regret = np.clip(np.nan_to_num(regret), 0.0, 1.0)

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    normalized = np.divide(
        work,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=geometric_scale > 0.0,
    )

    row_ratio = np.divide(
        work,
        row_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=row_scale > 0.0,
    )
    col_ratio = np.divide(
        work,
        col_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=col_scale > 0.0,
    )

    distance_quality = np.exp(
        -np.clip(1.08 * normalized, 0.0, 60.0)
    )

    endpoint_quality = np.exp(
        -np.clip(0.5 * (row_ratio + col_ratio), 0.0, 60.0)
    )

    ranks = np.argsort(
        np.argsort(work, axis=1, kind="stable"),
        axis=1,
    ).astype(float)

    rank_quality = np.exp(
        -ranks / max(1.0, 0.90 * np.sqrt(float(n)))
    )
    bidirectional_rank = np.sqrt(
        rank_quality * rank_quality.T
    )

    nearest_index = np.argmin(work, axis=1)
    indices = np.arange(n)
    mutual_nearest = (
        (nearest_index[:, None] == indices[None, :])
        & (nearest_index[None, :] == indices[:, None])
        & valid
    )
    mutual_bonus = np.where(mutual_nearest, 1.16, 1.0)

    urgency = 1.0 + 0.16 * (
        regret[:, None] + regret[None, :]
    )

    proximity = np.exp(
        -np.clip(normalized, 0.0, 40.0)
    )
    proximity[~valid] = 0.0
    np.fill_diagonal(proximity, 0.0)

    candidate_mask = valid & (
        ranks < float(k)
    )
    candidate_affinity = proximity * candidate_mask

    continuation = np.zeros((n, n), dtype=float)
    for h in range(n):
        through_h = np.minimum(
            candidate_affinity[:, h][:, None],
            candidate_affinity[h, :][None, :],
        )
        continuation = np.maximum(continuation, through_h)

    continuation[~valid] = 0.0
    continuation_scale = max(float(np.max(continuation)), eps)
    continuation /= continuation_scale

    local_support = candidate_affinity.sum(axis=1)
    support_median = float(np.median(local_support[local_support > 0.0]))
    if not np.isfinite(support_median) or support_median <= 0.0:
        support_median = 1.0

    rarity = np.sqrt(
        support_median / np.maximum(local_support, eps)
    )
    rarity = np.clip(rarity, 0.75, 1.50)
    rarity_pair = 0.5 * (
        rarity[:, None] + rarity[None, :]
    )

    bridge_factor = np.clip(
        1.0
        + 0.06 * (rarity_pair - 1.0)
        + 0.08 * continuation * endpoint_quality,
        0.90,
        1.10,
    )

    rank_factor = (
        0.54
        + 0.28 * bidirectional_rank
        + 0.18 * endpoint_quality
    )

    scores = (
        distance_quality
        * rank_factor
        * urgency
        * mutual_bonus
        * bridge_factor
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    if np.allclose(d, d.T, equal_nan=True):
        scores = 0.5 * (scores + scores.T)

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
