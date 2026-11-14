import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank-based, lookahead-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    masked = np.where(valid, d, np.inf)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    finite_row_best = row_best[np.isfinite(row_best)]
    finite_col_best = col_best[np.isfinite(col_best)]

    fallback = (
        float(np.median(d[valid]))
        if np.any(valid)
        else 1.0
    )
    row_reference = np.where(
        np.isfinite(row_best),
        row_best,
        fallback,
    )
    col_reference = np.where(
        np.isfinite(col_best),
        col_best,
        fallback,
    )

    row_order = np.argsort(masked, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=int)
    row_index = np.arange(n)[:, None]
    row_rank[row_index, row_order] = np.arange(n)[None, :]

    col_order = np.argsort(masked, axis=0, kind="stable")
    col_rank = np.empty((n, n), dtype=int)
    col_index = np.arange(n)[None, :]
    col_rank[col_order, col_index] = np.arange(n)[:, None]

    usable_rank = np.minimum(row_rank, 7)
    rank_weight = 1.0 / (1.0 + usable_rank.astype(float))
    rank_weight[~valid] = 0.0

    row_excess = np.maximum(
        np.where(valid, d, row_reference[:, None]) - row_reference[:, None],
        0.0,
    )
    col_excess = np.maximum(
        np.where(valid, d, col_reference[None, :]) - col_reference[None, :],
        0.0,
    )

    row_scale = np.maximum(
        np.partition(
            np.where(valid, row_excess, np.inf),
            min(1, n - 1),
            axis=1,
        )[:, min(1, n - 1)],
        eps,
    )
    col_scale = np.maximum(
        np.partition(
            np.where(valid, col_excess, np.inf),
            min(1, n - 1),
            axis=0,
        )[min(1, n - 1), :],
        eps,
    )

    row_regret = np.divide(
        row_excess,
        row_scale[:, None],
        out=np.zeros_like(d),
        where=valid,
    )
    col_regret = np.divide(
        col_excess,
        col_scale[None, :],
        out=np.zeros_like(d),
        where=valid,
    )

    row_regret_bonus = 1.0 / (1.0 + row_regret)
    col_regret_bonus = 1.0 / (1.0 + col_regret)

    successor_options = valid.sum(axis=1).astype(float)
    predecessor_options = valid.sum(axis=0).astype(float)

    successor_access = np.divide(
        successor_options,
        successor_options.max(initial=1.0),
    )
    predecessor_access = np.divide(
        predecessor_options,
        predecessor_options.max(initial=1.0),
    )

    target_access = np.sqrt(
        successor_access[None, :] * predecessor_access[None, :]
    )

    next_best = np.min(
        np.where(valid, masked, np.inf),
        axis=1,
    )
    next_best = np.where(np.isfinite(next_best), next_best, fallback)

    lookahead = np.zeros_like(d)
    for j in range(n):
        continuation = np.where(valid[j], d[j], np.inf)
        continuation[j] = np.inf
        continuation_best = np.min(continuation)
        if not np.isfinite(continuation_best):
            continuation_best = next_best[j]
        lookahead[:, j] = continuation_best

    lookahead_scale = max(float(np.median(np.abs(lookahead[valid]))), eps)
    lookahead_bonus = 1.0 / (
        1.0 + np.divide(
            lookahead,
            lookahead_scale,
            out=np.zeros_like(d),
            where=valid,
        )
    )

    column_pressure = np.sum(rank_weight * valid, axis=0)
    column_pressure = np.maximum(column_pressure, 1.0)
    destination_bonus = np.divide(
        1.0,
        column_pressure[None, :],
        out=np.zeros_like(d),
        where=valid,
    )

    score = (
        0.38 * rank_weight
        + 0.22 * row_regret_bonus
        + 0.16 * col_regret_bonus
        + 0.14 * lookahead_bonus
        + 0.07 * target_access
        + 0.03 * destination_bonus
    )

    score[~valid] = 0.0

    row_totals = score.sum(axis=1, keepdims=True)
    normalized = np.divide(
        score,
        row_totals,
        out=np.zeros_like(score),
        where=row_totals > 0.0,
    )

    concentration = np.sum(normalized * normalized, axis=0)
    crowding = 1.0 / (1.0 + concentration[None, :])
    normalized *= crowding
    normalized[~valid] = 0.0

    maximum = float(np.max(normalized))
    if maximum > 0.0 and np.isfinite(maximum):
        normalized /= maximum

    normalized[~np.isfinite(normalized)] = 0.0
    np.fill_diagonal(normalized, 0.0)
    return normalized.astype(float, copy=False)
