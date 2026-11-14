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
    sorted_rows = np.sort(work, axis=1)
    sorted_cols = np.sort(work, axis=0)

    k = min(5, n - 1)

    row_nearest = sorted_rows[:, :k]
    col_nearest = sorted_cols[:k, :]

    row_counts = np.isfinite(row_nearest).sum(axis=1)
    col_counts = np.isfinite(col_nearest).sum(axis=0)

    row_safe = np.where(np.isfinite(row_nearest), row_nearest, 0.0)
    col_safe = np.where(np.isfinite(col_nearest), col_nearest, 0.0)

    row_scale = np.divide(
        row_safe.sum(axis=1),
        np.maximum(row_counts, 1),
        out=np.ones(n, dtype=float),
        where=row_counts > 0,
    )
    col_scale = np.divide(
        col_safe.sum(axis=0),
        np.maximum(col_counts, 1),
        out=np.ones(n, dtype=float),
        where=col_counts > 0,
    )

    row_scale = np.maximum(row_scale, np.finfo(float).eps)
    col_scale = np.maximum(col_scale, np.finfo(float).eps)

    geometric_scale = np.sqrt(row_scale[:, None] * col_scale[None, :])
    normalized_distance = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    finite_norm = normalized_distance[valid]
    global_scale = np.median(finite_norm) if finite_norm.size else 1.0
    global_scale = max(float(global_scale), np.finfo(float).eps)

    row_ratio = np.divide(
        d,
        row_scale[:, None],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        col_scale[None, :],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    row_quality = np.exp(-np.clip(row_ratio, 0.0, 50.0) / 0.95)
    col_quality = np.exp(-np.clip(col_ratio, 0.0, 50.0) / 0.95)
    mutual_quality = np.sqrt(row_quality * col_quality)

    row_order = np.argsort(np.argsort(work, axis=1), axis=1)
    col_order = np.argsort(np.argsort(work, axis=0), axis=0)

    row_rank = np.zeros_like(d, dtype=float)
    col_rank = np.zeros_like(d, dtype=float)
    row_rank[valid] = row_order[valid] + 1.0
    col_rank[valid] = col_order[valid] + 1.0

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.12 * row_rank) * (1.0 + 0.12 * col_rank)
    )

    row_best = sorted_rows[:, 0]
    row_second = sorted_rows[:, 1] if n > 2 else row_best.copy()
    col_best = sorted_cols[0, :]
    col_second = sorted_cols[1, :] if n > 2 else col_best.copy()

    row_best = np.where(np.isfinite(row_best), row_best, row_scale)
    row_second = np.where(np.isfinite(row_second), row_second, 2.0 * row_best)
    col_best = np.where(np.isfinite(col_best), col_best, col_scale)
    col_second = np.where(np.isfinite(col_second), col_second, 2.0 * col_best)

    row_regret = np.clip(
        (row_second - row_best) / np.maximum(row_second, np.finfo(float).eps),
        0.0,
        1.0,
    )
    col_regret = np.clip(
        (col_second - col_best) / np.maximum(col_second, np.finfo(float).eps),
        0.0,
        1.0,
    )

    row_dispersion = np.clip(row_regret, 0.0, 1.0)
    row_mix = 0.68 - 0.18 * row_dispersion

    distance_affinity = np.exp(
        -np.clip(normalized_distance / global_scale, 0.0, 50.0) / 1.05
    )
    rank_affinity = rank_quality * mutual_quality

    score = (
        row_mix[:, None] * distance_affinity
        + (1.0 - row_mix[:, None]) * rank_affinity
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

    endpoint_scarcity = 1.0 / np.sqrt(
        np.maximum(row_competition[:, None], 1.0)
        * np.maximum(col_competition[None, :], 1.0)
    )

    soft_bridge = 1.0 + 0.08 * endpoint_scarcity * mutual_quality
    score *= soft_bridge

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_valid_count = valid.sum(axis=1, keepdims=True)
    floor = np.divide(
        0.015,
        np.maximum(row_valid_count, 1),
        out=np.zeros_like(row_valid_count, dtype=float),
        where=row_valid_count > 0,
    )
    score = np.where(valid, score + floor, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )
