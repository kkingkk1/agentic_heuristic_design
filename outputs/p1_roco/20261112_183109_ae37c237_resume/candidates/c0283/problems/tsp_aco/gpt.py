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

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not valid.any():
        return result

    inf = np.inf
    eps = np.finfo(float).eps
    work = np.where(valid, d, inf)

    k = min(4, n - 1)
    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    finite_nearest = np.isfinite(nearest)
    neighbor_count = finite_nearest.sum(axis=1)

    neighbor_sum = np.where(finite_nearest, nearest, 0.0).sum(axis=1)
    local_scale = np.divide(
        neighbor_sum,
        np.maximum(neighbor_count, 1),
        out=np.ones(n, dtype=float),
        where=neighbor_count > 0,
    )
    local_scale = np.maximum(local_scale, eps)

    ordered = np.sort(work, axis=1)
    best = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)

    regret = np.clip(
        (safe_second - safe_best) / np.maximum(safe_second, eps),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.16 * regret

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    normalized = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, inf),
        where=valid,
    )
    distance_quality = np.exp(-np.minimum(1.05 * normalized, 50.0))

    row_ratio = np.divide(
        d,
        row_scale,
        out=np.full_like(d, inf),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        col_scale,
        out=np.full_like(d, inf),
        where=valid,
    )

    row_quality = np.exp(-np.minimum(row_ratio, 50.0))
    col_quality = np.exp(-np.minimum(col_ratio, 50.0))
    mutual_quality = np.sqrt(row_quality * col_quality)

    row_order = np.argsort(work, axis=1, kind="stable")
    row_ranks = np.empty_like(row_order, dtype=float)
    row_ranks[row_order] = np.arange(1, n + 1, dtype=float)

    col_order = np.argsort(work, axis=0, kind="stable")
    col_ranks = np.empty_like(col_order, dtype=float)
    col_ranks[col_order] = np.arange(1, n + 1, dtype=float)

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.10 * row_ranks)
        * (1.0 + 0.10 * col_ranks)
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

    score = (
        distance_quality
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
        out=result,
        where=totals > 0.0,
    )
