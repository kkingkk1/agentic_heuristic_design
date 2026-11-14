import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed, scale-invariant, row-normalized TSP edge priors."""
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
    work = np.where(valid, d, np.inf)

    positive = d[valid & (d > 0.0)]
    fallback_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    fallback_scale = max(fallback_scale, eps)

    def endpoint_statistics(matrix: np.ndarray):
        ordered = np.sort(matrix, axis=1)
        k = min(3, n - 1)

        neighborhood = ordered[:, :k]
        finite = np.isfinite(neighborhood)
        safe = np.where(finite, neighborhood, 0.0)

        counts = finite.sum(axis=1)
        scale = np.divide(
            safe.sum(axis=1),
            np.maximum(counts, 1),
            out=np.full(n, fallback_scale, dtype=float),
            where=counts > 0,
        )
        scale = np.maximum(scale, eps)

        best = ordered[:, 0]
        second = ordered[:, 1] if n > 2 else best.copy()

        safe_best = np.where(np.isfinite(best), best, scale)
        safe_second = np.where(
            np.isfinite(second),
            second,
            2.0 * np.maximum(safe_best, eps),
        )

        regret = np.divide(
            safe_second - safe_best,
            np.maximum(safe_second, eps),
            out=np.zeros(n, dtype=float),
            where=np.isfinite(safe_second),
        )
        regret = np.clip(regret, 0.0, 1.0)

        return scale, regret, safe_second

    out_scale, out_regret, out_second = endpoint_statistics(work)
    in_scale, in_regret, in_second = endpoint_statistics(work.T)

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
        -np.clip(0.95 * relative_distance, 0.0, 60.0)
    )

    out_ratio = np.divide(
        d,
        out_scale[:, None],
        out=np.full_like(d, np.inf),
        where=valid,
    )
    in_ratio = np.divide(
        d,
        in_scale[None, :],
        out=np.full_like(d, np.inf),
        where=valid,
    )

    out_quality = np.exp(-np.clip(out_ratio, 0.0, 60.0))
    in_quality = np.exp(-np.clip(in_ratio, 0.0, 60.0))
    bilateral_quality = np.sqrt(out_quality * in_quality)

    row_order = np.argsort(work, axis=1, kind="stable")
    row_rank = np.full((n, n), np.inf, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    col_order = np.argsort(work, axis=0, kind="stable")
    col_rank = np.full((n, n), np.inf, dtype=float)
    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    rank_scale = max(1.0, 0.9 * np.sqrt(float(n)))
    rank_quality = np.exp(
        -0.5 * (row_rank + col_rank) / rank_scale
    )
    rank_quality = np.where(valid, rank_quality, 0.0)

    outgoing_competition = np.sum(
        valid & (d <= out_second[:, None]),
        axis=1,
        dtype=float,
    )
    incoming_competition = np.sum(
        valid & (d <= in_second[None, :]),
        axis=0,
        dtype=float,
    )

    scarcity = 1.0 + 0.12 / np.sqrt(
        np.maximum(outgoing_competition[:, None], 1.0)
        * np.maximum(incoming_competition[None, :], 1.0)
    )

    urgency = 1.0 + 0.22 * (
        out_regret[:, None] + in_regret[None, :]
    )

    nearest_bonus = np.ones((n, n), dtype=float)
    nearest_bonus[row_rank == 0.0] *= 1.08
    nearest_bonus[col_rank == 0.0] *= 1.08

    long_edge_penalty = np.exp(
        -0.35 * np.maximum(relative_distance - 1.0, 0.0)
    )

    preference = (
        0.48
        + 0.32 * bilateral_quality
        + 0.20 * rank_quality
    )

    scores = (
        distance_quality
        * preference
        * scarcity
        * urgency
        * nearest_bonus
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

    totals = scores.sum(axis=1, keepdims=True)

    return np.divide(
        scores,
        totals,
        out=np.zeros_like(scores),
        where=totals > 0.0,
    )
