import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros_like(d, dtype=float)

    if n < 2:
        return score

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return score

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    def endpoint_statistics(matrix: np.ndarray):
        ordered = np.sort(matrix, axis=1)
        best = ordered[:, 0]

        if n > 2:
            second = ordered[:, 1]
        else:
            second = best.copy()

        finite_best = np.isfinite(best)
        safe_best = np.where(finite_best, best, 1.0)
        safe_second = np.where(
            np.isfinite(second),
            second,
            2.0 * safe_best,
        )

        local_scale = np.maximum(
            0.5 * (safe_best + safe_second),
            eps,
        )

        regret = np.clip(
            (safe_second - safe_best)
            / np.maximum(safe_second, eps),
            0.0,
            1.0,
        )

        return local_scale, regret

    outgoing_scale, outgoing_regret = endpoint_statistics(work)
    incoming_scale, incoming_regret = endpoint_statistics(work.T)

    pair_scale = np.sqrt(
        outgoing_scale[:, None] * incoming_scale[None, :]
    )

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf),
        where=valid & (pair_scale > 0.0),
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

    ranks = np.full((n, n), np.inf, dtype=float)

    row_order = np.argsort(work, axis=1, kind="stable")
    row_rank = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], row_order] = row_rank[None, :]

    col_order = np.argsort(work, axis=0, kind="stable")
    ranks_in = np.full((n, n), np.inf, dtype=float)
    ranks_in[col_order, np.arange(n)[None, :]] = row_rank[:, None]

    rank_scale = max(1.0, np.sqrt(float(n)))
    neighborhood_term = np.exp(
        -0.5 * (ranks + ranks_in) / rank_scale
    )
    neighborhood_term = np.nan_to_num(
        neighborhood_term,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    regret_term = 1.0 + 0.45 * (
        outgoing_regret[:, None]
        + incoming_regret[None, :]
    )

    continuation = np.sqrt(
        np.clip(
            np.max(
                np.where(valid, outgoing_quality, 0.0),
                axis=1,
            )[:, None],
            0.0,
            1.0,
        )
        * np.clip(
            np.max(
                np.where(valid, incoming_quality, 0.0),
                axis=0,
            )[None, :],
            0.0,
            1.0,
        )
    )

    long_edge_penalty = np.exp(
        -0.30 * np.maximum(relative_distance - 1.0, 0.0)
    )

    score = (
        distance_term
        * (0.55 + 0.25 * mutual_quality + 0.20 * neighborhood_term)
        * regret_term
        * long_edge_penalty
        * (1.0 + 0.06 * continuation)
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
