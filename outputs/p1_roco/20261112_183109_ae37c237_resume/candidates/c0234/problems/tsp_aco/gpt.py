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
    counts = finite_nearest.sum(axis=1)

    local_sum = np.where(finite_nearest, nearest, 0.0).sum(axis=1)
    local_scale = np.divide(
        local_sum,
        np.maximum(counts, 1),
        out=np.ones(n, dtype=float),
        where=counts > 0,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).tiny)

    best = np.min(work, axis=1)
    if n > 2:
        second = np.partition(work, kth=1, axis=1)[:, 1]
    else:
        second = best.copy()

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(
        np.isfinite(second),
        second,
        np.maximum(2.0 * safe_best, np.finfo(float).tiny),
    )

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, np.finfo(float).tiny),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.20 * regret

    row_scale_log = np.log(local_scale)[:, None]
    col_scale_log = np.log(local_scale)[None, :]
    geometric_scale_log = 0.5 * (row_scale_log + col_scale_log)

    safe_distance = np.maximum(d, np.finfo(float).tiny)
    normalized_log_distance = (
        np.log(safe_distance) - geometric_scale_log
    )
    normalized_distance = np.exp(
        np.clip(normalized_log_distance, -50.0, 50.0)
    )

    row_log_distance = np.log(safe_distance) - row_scale_log
    col_log_distance = np.log(safe_distance) - col_scale_log

    row_ratio = np.exp(np.clip(row_log_distance, -50.0, 50.0))
    col_ratio = np.exp(np.clip(col_log_distance, -50.0, 50.0))

    row_quality_log = -np.clip(row_ratio, 0.0, 50.0)
    col_quality_log = -np.clip(col_ratio, 0.0, 50.0)
    mutual_quality_log = 0.5 * (row_quality_log + col_quality_log)

    row_order = np.argsort(
        np.where(valid, d, np.inf),
        axis=1,
        kind="stable",
    )
    row_rank_matrix = np.argsort(
        row_order,
        axis=1,
        kind="stable",
    ).astype(float) + 1.0

    col_order = np.argsort(
        np.where(valid, d, np.inf),
        axis=0,
        kind="stable",
    )
    col_rank_matrix = np.argsort(
        col_order,
        axis=0,
        kind="stable",
    ).astype(float) + 1.0

    row_ranks = np.where(valid, row_rank_matrix, 0.0)
    col_ranks = np.where(valid, col_rank_matrix, 0.0)

    rank_quality_log = -0.5 * (
        np.log1p(0.12 * row_ranks)
        + np.log1p(0.12 * col_ranks)
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
        -np.clip(normalized_distance, 0.0, 50.0)
        + mutual_quality_log
        + rank_quality_log
        + np.log(scarcity)
        + np.log(urgency[:, None])
        + np.log(urgency[None, :])
    )

    score = np.exp(np.clip(log_score, -745.0, 50.0))
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    symmetric_input = (
        np.array_equal(valid, valid.T)
        and np.array_equal(
            np.where(valid, d, 0.0),
            np.where(valid.T, d.T, 0.0),
        )
    )
    if symmetric_input:
        score = 0.5 * (score + score.T)
        score[~valid] = 0.0
        np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
