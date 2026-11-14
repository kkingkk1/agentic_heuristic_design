import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed, row-normalized edge-priority indicators for TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n < 2:
        return score

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    def endpoint_stats(matrix: np.ndarray):
        ordered = np.sort(matrix, axis=1)
        best = ordered[:, 0]

        if n > 2:
            second = ordered[:, 1]
            third = ordered[:, 2]
        else:
            second = best.copy()
            third = second.copy()

        finite_best = np.isfinite(best)
        safe_best = np.where(finite_best, best, 1.0)
        safe_second = np.where(
            np.isfinite(second),
            second,
            2.0 * safe_best,
        )
        safe_third = np.where(
            np.isfinite(third),
            third,
            2.0 * safe_second,
        )

        local_scale = np.maximum(
            (safe_best + safe_second + safe_third) / 3.0,
            eps,
        )

        regret = np.clip(
            (safe_second - safe_best)
            / np.maximum(safe_second, eps),
            0.0,
            1.0,
        )

        return local_scale, regret

    out_scale, out_regret = endpoint_stats(work)
    in_scale, in_regret = endpoint_stats(work.T)

    pair_scale = np.sqrt(
        np.maximum(out_scale[:, None], eps)
        * np.maximum(in_scale[None, :], eps)
    )

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf),
        where=valid,
    )

    distance_quality = np.exp(
        -np.clip(relative_distance, 0.0, 60.0)
    )

    out_ratio = np.divide(
        d,
        np.maximum(out_scale[:, None], eps),
        out=np.full_like(d, np.inf),
        where=valid,
    )
    in_ratio = np.divide(
        d,
        np.maximum(in_scale[None, :], eps),
        out=np.full_like(d, np.inf),
        where=valid,
    )

    out_quality = np.exp(-np.clip(out_ratio, 0.0, 60.0))
    in_quality = np.exp(-np.clip(in_ratio, 0.0, 60.0))
    mutual_quality = np.sqrt(out_quality * in_quality)

    row_order = np.argsort(work, axis=1, kind="stable")
    row_ranks = np.full((n, n), np.inf, dtype=float)
    row_ranks[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    col_order = np.argsort(work, axis=0, kind="stable")
    col_ranks = np.full((n, n), np.inf, dtype=float)
    col_ranks[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    rank_scale = max(1.0, np.sqrt(float(n)))
    bidirectional_rank = np.exp(
        -0.5 * (row_ranks + col_ranks) / rank_scale
    )
    bidirectional_rank = np.nan_to_num(
        bidirectional_rank,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    regret_bonus = 1.0 + 0.30 * (
        out_regret[:, None] + in_regret[None, :]
    )

    long_edge_penalty = np.exp(
        -0.35 * np.maximum(relative_distance - 1.0, 0.0)
    )

    combined_preference = (
        0.50
        + 0.30 * mutual_quality
        + 0.20 * bidirectional_rank
    )

    score = (
        distance_quality
        * combined_preference
        * regret_bonus
        * long_edge_penalty
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
        out=np.zeros_like(score),
        where=row_totals > 0.0,
    )
