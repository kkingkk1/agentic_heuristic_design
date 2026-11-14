import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    eps = np.finfo(float).eps
    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    finite_values = d[valid]
    global_scale = float(np.median(finite_values))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        positive = finite_values[finite_values > 0.0]
        global_scale = (
            float(np.median(positive)) if positive.size else 1.0
        )
    global_scale = max(global_scale, eps)

    work = np.where(valid, d, np.inf)

    k = min(4, n - 1)
    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    nearest_valid = np.isfinite(nearest)
    nearest_count = nearest_valid.sum(axis=1)
    nearest_sum = np.sum(
        np.where(nearest_valid, nearest, 0.0),
        axis=1,
    )

    local_scale = np.divide(
        nearest_sum,
        np.maximum(nearest_count, 1),
        out=np.full(n, global_scale, dtype=float),
        where=nearest_count > 0,
    )
    local_scale = np.maximum(local_scale, eps)

    ordered = np.sort(work, axis=1)
    best = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, global_scale)
    safe_second = np.where(
        np.isfinite(second),
        second,
        2.0 * safe_best,
    )

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, eps),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.14 * regret

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    normalized = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    distance_quality = np.exp(
        -np.clip(1.08 * normalized, 0.0, 50.0)
    )

    row_ratio = np.divide(
        d,
        row_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        col_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    row_quality = np.exp(-np.clip(row_ratio, 0.0, 50.0))
    col_quality = np.exp(-np.clip(col_ratio, 0.0, 50.0))
    mutual_quality = np.sqrt(row_quality * col_quality)

    row_order = np.argsort(
        np.where(valid, d, np.inf),
        axis=1,
        kind="stable",
    )
    row_rank_order = np.argsort(row_order, axis=1, kind="stable")
    row_ranks = np.where(valid, row_rank_order + 1.0, 0.0)

    col_order = np.argsort(
        np.where(valid, d, np.inf),
        axis=0,
        kind="stable",
    )
    col_rank_order = np.argsort(col_order, axis=0, kind="stable")
    col_ranks = np.where(valid, col_rank_order + 1.0, 0.0)

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.07 * row_ranks)
        * (1.0 + 0.07 * col_ranks)
    )

    row_competition = np.sum(
        valid & (d <= safe_second[:, None]),
        axis=1,
        dtype=float,
    )
    col_competition = np.sum(
        valid & (d <= safe_second[None, :]),
        axis=0,
        dtype=float,
    )

    scarcity = 1.0 + 0.06 / np.sqrt(
        np.maximum(row_competition[:, None], 1.0)
        * np.maximum(col_competition[None, :], 1.0)
    )

    reverse_valid = valid.T
    reverse_normalized = np.divide(
        d.T,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=reverse_valid,
    )
    reverse_quality = np.exp(
        -np.clip(0.92 * reverse_normalized, 0.0, 50.0)
    )

    symmetry_quality = np.where(
        valid & reverse_valid,
        np.sqrt(distance_quality * reverse_quality),
        distance_quality,
    )

    score = (
        symmetry_quality
        * mutual_quality
        * rank_quality
        * scarcity
        * urgency[:, None]
        * urgency[None, :]
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)

    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )
