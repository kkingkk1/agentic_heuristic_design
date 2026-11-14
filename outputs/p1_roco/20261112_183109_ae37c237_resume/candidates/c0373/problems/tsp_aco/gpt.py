import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return stochastic edge-priority indicators for directed or symmetric TSP."""
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

    inf = np.inf
    eps = np.finfo(float).eps
    work = np.where(valid, d, inf)

    row_sorted = np.sort(work, axis=1)
    col_sorted = np.sort(work, axis=0)

    k = min(4, n - 1)
    row_local = row_sorted[:, :k]
    col_local = col_sorted[:k, :]

    row_counts = np.isfinite(row_local).sum(axis=1)
    col_counts = np.isfinite(col_local).sum(axis=0)

    row_sum = np.where(np.isfinite(row_local), row_local, 0.0).sum(axis=1)
    col_sum = np.where(np.isfinite(col_local), col_local, 0.0).sum(axis=0)

    row_scale = np.divide(
        row_sum,
        np.maximum(row_counts, 1),
        out=np.ones(n, dtype=float),
        where=row_counts > 0,
    )
    col_scale = np.divide(
        col_sum,
        np.maximum(col_counts, 1),
        out=np.ones(n, dtype=float),
        where=col_counts > 0,
    )

    row_scale = np.maximum(row_scale, eps)
    col_scale = np.maximum(col_scale, eps)

    row_best = row_sorted[:, 0]
    col_best = col_sorted[0, :]

    row_second = row_sorted[:, 1] if n > 2 else row_best.copy()
    col_second = col_sorted[1, :] if n > 2 else col_best.copy()

    row_second = np.where(np.isfinite(row_second), row_second, 2.0 * row_best)
    col_second = np.where(np.isfinite(col_second), col_second, 2.0 * col_best)

    safe_row_best = np.where(np.isfinite(row_best), row_best, row_scale)
    safe_col_best = np.where(np.isfinite(col_best), col_best, col_scale)

    row_regret = np.clip(
        (row_second - safe_row_best)
        / np.maximum(row_second, eps),
        0.0,
        1.0,
    )
    col_regret = np.clip(
        (col_second - safe_col_best)
        / np.maximum(col_second, eps),
        0.0,
        1.0,
    )

    row_urgency = 1.0 + 0.10 * row_regret
    col_urgency = 1.0 + 0.10 * col_regret

    geometric_scale = np.sqrt(row_scale[:, None] * col_scale[None, :])
    normalized_distance = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, inf, dtype=float),
        where=valid,
    )

    distance_quality = np.exp(
        -np.clip(1.10 * normalized_distance, 0.0, 50.0)
    )

    row_ratio = np.divide(
        d,
        row_scale[:, None],
        out=np.full_like(d, inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        col_scale[None, :],
        out=np.full_like(d, inf, dtype=float),
        where=valid,
    )

    row_quality = np.exp(-np.clip(row_ratio, 0.0, 50.0))
    col_quality = np.exp(-np.clip(col_ratio, 0.0, 50.0))
    mutual_quality = np.sqrt(row_quality * col_quality)

    row_order = np.argsort(np.where(valid, d, inf), axis=1)
    col_order = np.argsort(np.where(valid, d, inf), axis=0)

    row_rank = np.zeros_like(d, dtype=float)
    col_rank = np.zeros_like(d, dtype=float)

    row_rank[row_order, np.arange(n)[:, None]] = np.arange(1, n + 1)
    col_rank[np.arange(n)[:, None], col_order] = np.arange(1, n + 1)[:, None]

    row_rank[~valid] = 0.0
    col_rank[~valid] = 0.0

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.07 * row_rank)
        * (1.0 + 0.07 * col_rank)
    )

    row_competition = np.sum(
        valid & (d <= row_second[:, None]),
        axis=1,
        dtype=float,
    )
    col_competition = np.sum(
        valid & (d <= col_second[None, :]),
        axis=0,
        dtype=float,
    )

    mutual_nearest = (
        valid
        & (d <= safe_row_best[:, None] * (1.0 + 1e-12))
        & (d <= safe_col_best[None, :] * (1.0 + 1e-12))
    )

    mutual_bonus = 1.0 + 0.06 * mutual_nearest.astype(float)

    scarcity_bonus = 1.0 + 0.035 / np.sqrt(
        np.maximum(row_competition[:, None], 1.0)
        * np.maximum(col_competition[None, :], 1.0)
    )

    endpoint_regret = np.sqrt(
        (1.0 + 0.10 * row_regret[:, None])
        * (1.0 + 0.10 * col_regret[None, :])
    )

    outgoing_after_edge = np.where(
        valid,
        np.minimum(row_second[:, None], row_scale[:, None] * 2.0),
        inf,
    )
    incoming_before_edge = np.where(
        valid,
        np.minimum(col_second[None, :], col_scale[None, :] * 2.0),
        inf,
    )

    continuation_scale = np.sqrt(row_scale[:, None] * col_scale[None, :])
    continuation_cost = (
        outgoing_after_edge + incoming_before_edge
    ) / np.maximum(2.0 * continuation_scale, eps)

    continuation_potential = 1.0 + 0.025 * np.exp(
        -np.clip(continuation_cost, 0.0, 20.0)
    )

    score = (
        distance_quality
        * mutual_quality
        * rank_quality
        * mutual_bonus
        * scarcity_bonus
        * endpoint_regret
        * continuation_potential
        * row_urgency[:, None]
        * col_urgency[None, :]
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )

    for i in range(n):
        candidates = valid[i]
        count = int(candidates.sum())

        if count == 0:
            continue

        uniform = np.zeros(n, dtype=float)
        uniform[candidates] = 1.0 / count

        concentration = float(probabilities[i].max())
        baseline = 1.0 / count
        if count == 1:
            mix = 0.0
        else:
            relative_concentration = np.clip(
                (concentration - baseline) / max(1.0 - baseline, eps),
                0.0,
                1.0,
            )
            mix = 0.04 + 0.12 * relative_concentration

        probabilities[i] = (
            (1.0 - mix) * probabilities[i]
            + mix * uniform
        )

        probabilities[i, ~candidates] = 0.0

    return probabilities
