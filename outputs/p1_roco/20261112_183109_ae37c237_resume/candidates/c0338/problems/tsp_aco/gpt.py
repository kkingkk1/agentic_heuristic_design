import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
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

    eps = np.finfo(float).eps
    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = float(np.max(finite_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    work = np.where(valid, d, np.inf)

    k = min(4, n - 1)
    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    nearest_valid = np.isfinite(nearest)

    nearest_sum = np.sum(
        np.where(nearest_valid, nearest, 0.0),
        axis=1,
    )
    nearest_count = np.sum(nearest_valid, axis=1)

    local_scale = np.divide(
        nearest_sum,
        np.maximum(nearest_count, 1),
        out=np.full(n, global_scale, dtype=float),
        where=nearest_count > 0,
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

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(
        np.isfinite(second),
        second,
        2.0 * safe_best,
    )

    regret = np.divide(
        safe_second - safe_best,
        np.maximum(safe_second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(safe_second),
    )
    regret = np.clip(regret, 0.0, 1.0)

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    normalized_distance = np.divide(
        d,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (geometric_scale > 0.0),
    )

    forward_quality = np.exp(
        -np.clip(1.15 * normalized_distance, 0.0, 60.0)
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
        -np.clip(0.34 * (row_ratio + col_ratio), 0.0, 60.0)
    )

    row_order = np.argsort(work, axis=1, kind="stable")
    row_rank_order = np.argsort(row_order, axis=1, kind="stable")
    row_ranks = np.where(valid, row_rank_order + 1.0, 0.0)

    reverse_work = work.T
    reverse_valid = valid.T
    reverse_normalized = np.divide(
        d.T,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=reverse_valid & (geometric_scale > 0.0),
    )
    reverse_quality = np.exp(
        -np.clip(1.15 * reverse_normalized, 0.0, 60.0)
    )

    symmetry_quality = np.where(
        valid & reverse_valid,
        np.sqrt(forward_quality * reverse_quality),
        forward_quality,
    )

    rank_quality = 1.0 / np.sqrt(
        1.0 + 0.075 * row_ranks
    )

    competition = np.sum(
        valid & (d <= safe_second[:, None]),
        axis=1,
        dtype=float,
    )
    scarcity = 1.0 + 0.06 / np.sqrt(
        np.maximum(competition[:, None], 1.0)
        * np.maximum(competition[None, :], 1.0)
    )

    urgency = 1.0 + 0.14 * regret

    nearest_neighbor = np.argmin(work, axis=1)
    indices = np.arange(n)
    mutual_nearest = (
        valid
        & reverse_valid
        & (nearest_neighbor[:, None] == indices[None, :])
        & (nearest_neighbor[None, :] == indices[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.18, 1.0)

    long_edge_penalty = np.exp(
        -0.22 * np.maximum(normalized_distance - 1.0, 0.0)
    )

    scores = (
        symmetry_quality
        * (
            0.62
            + 0.23 * endpoint_quality
            + 0.15 * rank_quality
        )
        * scarcity
        * urgency[:, None]
        * urgency[None, :]
        * mutual_bonus
        * long_edge_penalty
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    row_totals = scores.sum(axis=1, keepdims=True)

    return np.divide(
        scores,
        row_totals,
        out=np.zeros_like(scores),
        where=row_totals > 0.0,
    ).astype(float, copy=False)
