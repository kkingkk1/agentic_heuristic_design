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

    if not np.any(valid):
        return result

    work = np.where(valid, d, np.inf)

    k = min(3, n - 1)
    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    finite_nearest = np.isfinite(nearest)
    neighbor_count = finite_nearest.sum(axis=1)

    local_scale = np.divide(
        np.where(finite_nearest, nearest, 0.0).sum(axis=1),
        np.maximum(neighbor_count, 1),
        out=np.ones(n, dtype=float),
        where=neighbor_count > 0,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    first_two = np.partition(work, kth=min(1, n - 1), axis=1)[:, :2]
    best = first_two[:, 0]
    second = first_two[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, np.finfo(float).eps),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.20 * regret

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    normalized = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
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

    row_order = np.argsort(work, axis=1, kind="stable")
    col_order = np.argsort(work, axis=0, kind="stable")

    row_ranks = np.zeros_like(d, dtype=float)
    col_ranks = np.zeros_like(d, dtype=float)

    row_rank_values = np.arange(1.0, n + 1.0)[None, :]
    col_rank_values = np.arange(1.0, n + 1.0)[:, None]

    np.put_along_axis(row_ranks, row_order, row_rank_values, axis=1)
    np.put_along_axis(col_ranks, col_order, col_rank_values, axis=0)

    row_ranks[~valid] = 0.0
    col_ranks[~valid] = 0.0

    rank_log_quality = -0.5 * (
        np.log1p(0.12 * row_ranks) + np.log1p(0.12 * col_ranks)
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

    scarcity = 1.0 + 0.08 / np.sqrt(
        np.maximum(row_competition[:, None], 1.0)
        * np.maximum(col_competition[None, :], 1.0)
    )

    log_score = (
        -np.clip(normalized, 0.0, 50.0)
        -0.5 * np.clip(row_ratio, 0.0, 50.0)
        -0.5 * np.clip(col_ratio, 0.0, 50.0)
        + rank_log_quality
        + np.log(scarcity)
        + np.log(urgency[:, None])
        + np.log(urgency[None, :])
    )

    score = np.zeros_like(d, dtype=float)
    score[valid] = np.exp(np.clip(log_score[valid], -745.0, 50.0))

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
