import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    work = np.where(valid, d, np.inf)
    eps = np.finfo(float).eps

    def endpoint_statistics(matrix: np.ndarray):
        if n == 2:
            first = matrix[:, 0].copy()
            second = matrix[:, 1].copy()
        else:
            nearest = np.partition(matrix, kth=1, axis=1)[:, :2]
            nearest.sort(axis=1)
            first = nearest[:, 0]
            second = nearest[:, 1]

        finite_first = np.isfinite(first)
        safe_first = np.where(finite_first, first, 1.0)
        safe_second = np.where(
            np.isfinite(second),
            second,
            np.maximum(2.0 * safe_first, 1.0),
        )

        scale = np.maximum(
            0.5 * (safe_first + safe_second),
            eps,
        )

        regret = np.clip(
            (safe_second - safe_first)
            / np.maximum(safe_second, eps),
            0.0,
            1.0,
        )

        return scale, regret

    outgoing_scale, outgoing_regret = endpoint_statistics(work)
    incoming_scale, incoming_regret = endpoint_statistics(work.T)

    relative_out = np.divide(
        d,
        outgoing_scale[:, None],
        out=np.full_like(d, np.inf),
        where=valid,
    )
    relative_in = np.divide(
        d,
        incoming_scale[None, :],
        out=np.full_like(d, np.inf),
        where=valid,
    )

    pair_relative = np.sqrt(
        np.maximum(relative_out, 0.0)
        * np.maximum(relative_in, 0.0)
    )

    distance_term = np.exp(
        -np.clip(0.5 * (relative_out + relative_in), 0.0, 60.0)
    )

    pair_term = np.exp(
        -np.clip(pair_relative, 0.0, 60.0)
    )

    row_order = np.argsort(work, axis=1, kind="stable")
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    col_order = np.argsort(work, axis=0, kind="stable")
    col_rank = np.empty_like(col_order, dtype=float)
    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    rank_scale = max(1.0, np.sqrt(float(n)))
    rank_term = np.exp(
        -0.5 * (row_rank + col_rank) / rank_scale
    )

    rank_term[~valid] = 0.0

    regret_term = 1.0 + 0.35 * (
        outgoing_regret[:, None]
        + incoming_regret[None, :]
    )

    row_best = np.max(
        np.where(valid, np.exp(-np.clip(relative_out, 0.0, 60.0)), 0.0),
        axis=1,
        keepdims=True,
    )
    col_best = np.max(
        np.where(valid, np.exp(-np.clip(relative_in, 0.0, 60.0)), 0.0),
        axis=0,
        keepdims=True,
    )

    continuation_term = np.sqrt(row_best * col_best)

    score = (
        (0.45 * distance_term + 0.35 * pair_term + 0.20 * rank_term)
        * regret_term
        * (1.0 + 0.08 * continuation_term)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    row_totals = score.sum(axis=1, keepdims=True)

    return np.divide(
        score,
        row_totals,
        out=result,
        where=row_totals > 0.0,
    )
