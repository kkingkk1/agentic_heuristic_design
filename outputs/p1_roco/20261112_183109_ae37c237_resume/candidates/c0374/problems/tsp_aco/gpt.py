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

    positive_values = d[valid & (d > 0.0)]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    work = np.where(valid, d, np.inf)
    k = min(4, n - 1)

    row_sorted = np.sort(work, axis=1)
    col_sorted = np.sort(work, axis=0)

    row_local = row_sorted[:, :k]
    col_local = col_sorted[:k, :]

    row_counts = np.isfinite(row_local).sum(axis=1)
    col_counts = np.isfinite(col_local).sum(axis=0)

    row_sum = np.where(np.isfinite(row_local), row_local, 0.0).sum(axis=1)
    col_sum = np.where(np.isfinite(col_local), col_local, 0.0).sum(axis=0)

    row_scale = np.divide(
        row_sum,
        np.maximum(row_counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=row_counts > 0,
    )
    col_scale = np.divide(
        col_sum,
        np.maximum(col_counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=col_counts > 0,
    )

    row_scale = np.maximum(row_scale, np.finfo(float).eps)
    col_scale = np.maximum(col_scale, np.finfo(float).eps)

    row_best = row_sorted[:, 0]
    row_second = row_sorted[:, 1] if n > 2 else row_best.copy()
    col_best = col_sorted[0, :]
    col_second = col_sorted[1, :] if n > 2 else col_best.copy()

    row_best = np.where(np.isfinite(row_best), row_best, row_scale)
    row_second = np.where(np.isfinite(row_second), row_second, 2.0 * row_best)
    col_best = np.where(np.isfinite(col_best), col_best, col_scale)
    col_second = np.where(np.isfinite(col_second), col_second, 2.0 * col_best)

    row_regret = np.clip(
        (row_second - row_best)
        / np.maximum(row_second, np.finfo(float).eps),
        0.0,
        1.0,
    )
    col_regret = np.clip(
        (col_second - col_best)
        / np.maximum(col_second, np.finfo(float).eps),
        0.0,
        1.0,
    )

    row_valid_values = np.where(valid, d, np.nan)
    row_mean = np.nanmean(row_valid_values, axis=1)
    row_mean = np.where(np.isfinite(row_mean), row_mean, global_scale)

    row_sq_mean = np.nanmean(
        np.where(valid, d * d, np.nan),
        axis=1,
    )
    row_sq_mean = np.where(np.isfinite(row_sq_mean), row_sq_mean, row_mean**2)

    row_std = np.sqrt(np.maximum(row_sq_mean - row_mean**2, 0.0))
    row_cv = row_std / np.maximum(row_mean, np.finfo(float).eps)

    adaptive_decay = 0.92 + 0.28 * (
        row_cv / (1.0 + row_cv)
    )

    geometric_scale = np.sqrt(row_scale[:, None] * col_scale[None, :])
    arithmetic_scale = 0.5 * (
        row_scale[:, None] + col_scale[None, :]
    )

    geometric_ratio = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    arithmetic_ratio = np.divide(
        d,
        arithmetic_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    normalized_distance = (
        0.65 * geometric_ratio + 0.35 * arithmetic_ratio
    )

    distance_quality = np.exp(
        -np.clip(adaptive_decay[:, None] * normalized_distance, 0.0, 50.0)
    )

    row_quality = np.exp(
        -np.clip(
            np.divide(
                d,
                row_scale[:, None],
                out=np.full_like(d, np.inf, dtype=float),
                where=valid,
            ),
            0.0,
            50.0,
        )
    )
    col_quality = np.exp(
        -np.clip(
            np.divide(
                d,
                col_scale[None, :],
                out=np.full_like(d, np.inf, dtype=float),
                where=valid,
            ),
            0.0,
            50.0,
        )
    )

    mutual_support = np.sqrt(row_quality * col_quality)

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

    scarcity_bonus = 1.0 + 0.035 / np.sqrt(
        np.maximum(row_competition[:, None], 1.0)
        * np.maximum(col_competition[None, :], 1.0)
    )

    regret_bonus = (
        1.0
        + 0.07 * row_regret[:, None]
        + 0.04 * col_regret[None, :]
    )

    score = (
        distance_quality
        * (0.94 + 0.06 * mutual_support)
        * scarcity_bonus
        * regret_bonus
    )
    score[~valid] = 0.0

    row_means = np.divide(
        score.sum(axis=1),
        np.maximum(valid.sum(axis=1), 1),
        out=np.zeros(n, dtype=float),
        where=valid.sum(axis=1) > 0,
    )
    score += 0.025 * row_means[:, None] * valid

    score = np.where(valid, np.power(np.maximum(score, 0.0), 0.90), 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
