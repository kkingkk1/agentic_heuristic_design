import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n < 2:
        return score

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return score

    positive = d[valid & (d > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size > 0
        else 1.0
    )
    eps = np.finfo(float).eps
    floor = max(eps, eps * global_scale)

    work = np.where(valid, d, np.inf)

    def endpoint_statistics(matrix: np.ndarray):
        ordered = np.sort(matrix, axis=1)
        best = ordered[:, 0]

        if n > 2:
            second = ordered[:, 1]
        else:
            second = best.copy()

        best = np.where(np.isfinite(best), best, global_scale)
        second = np.where(np.isfinite(second), second, 2.0 * best)

        safe_best = np.maximum(best, floor)
        safe_second = np.maximum(second, floor)

        local_scale = 0.5 * (safe_best + safe_second)
        local_scale = np.maximum(local_scale, floor)

        regret = np.divide(
            safe_second - safe_best,
            safe_second,
            out=np.zeros_like(safe_best),
            where=safe_second > 0.0,
        )
        regret = np.clip(regret, 0.0, 1.0)

        return local_scale, regret

    outgoing_scale, outgoing_regret = endpoint_statistics(work)
    incoming_scale, incoming_regret = endpoint_statistics(work.T)

    pair_scale = np.sqrt(
        outgoing_scale[:, None] * incoming_scale[None, :]
    )
    pair_scale = np.maximum(pair_scale, floor)

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf),
        where=valid,
    )

    distance_term = np.exp(
        -np.clip(relative_distance, 0.0, 60.0)
    )

    outgoing_ratio = np.divide(
        d,
        outgoing_scale[:, None],
        out=np.full_like(d, np.inf),
        where=valid,
    )
    incoming_ratio = np.divide(
        d,
        incoming_scale[None, :],
        out=np.full_like(d, np.inf),
        where=valid,
    )

    outgoing_quality = np.exp(
        -np.clip(outgoing_ratio, 0.0, 60.0)
    )
    incoming_quality = np.exp(
        -np.clip(incoming_ratio, 0.0, 60.0)
    )

    mutual_quality = np.sqrt(
        outgoing_quality * incoming_quality
    )

    ranks_out = np.full((n, n), np.inf, dtype=float)
    row_order = np.argsort(work, axis=1, kind="stable")
    rank_values = np.arange(n, dtype=float)
    ranks_out[
        np.arange(n)[:, None],
        row_order
    ] = rank_values[None, :]

    ranks_in = np.full((n, n), np.inf, dtype=float)
    column_order = np.argsort(work, axis=0, kind="stable")
    ranks_in[
        column_order,
        np.arange(n)[None, :]
    ] = rank_values[:, None]

    rank_scale = max(1.0, np.sqrt(float(n)))
    neighborhood_term = np.exp(
        -0.5 * (ranks_out + ranks_in) / rank_scale
    )
    neighborhood_term = np.nan_to_num(
        neighborhood_term,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    regret_term = 1.0 + 0.14 * (
        outgoing_regret[:, None]
        + incoming_regret[None, :]
    )

    mutual_term = 0.88 + 0.08 * mutual_quality
    neighborhood_bonus = 1.0 + 0.08 * neighborhood_term

    score = (
        distance_term
        * mutual_term
        * neighborhood_bonus
        * regret_term
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
