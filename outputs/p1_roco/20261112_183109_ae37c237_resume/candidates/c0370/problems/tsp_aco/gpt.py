import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized heuristic indicators for stochastic TSP sampling."""
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

    max_k = min(4, n - 1)
    nearest = sorted_rows[:, :max_k]
    finite_nearest = np.isfinite(nearest)
    counts = finite_nearest.sum(axis=1)

    eps = np.finfo(float).eps
    first = nearest[:, 0]
    safe_first = np.where(np.isfinite(first), first, 1.0)

    if max_k >= 4:
        fourth = nearest[:, 3]
        finite_fourth = np.isfinite(fourth)

        spread = np.divide(
            fourth - safe_first,
            np.maximum(safe_first, eps),
            out=np.full(n, np.inf, dtype=float),
            where=finite_fourth,
        )

        adaptive_k = np.where(spread <= 0.75, 4, 2)
        adaptive_k = np.minimum(adaptive_k, np.maximum(counts, 1))
    elif max_k == 3:
        third = nearest[:, 2]
        finite_third = np.isfinite(third)

        spread = np.divide(
            third - safe_first,
            np.maximum(safe_first, eps),
            out=np.full(n, np.inf, dtype=float),
            where=finite_third,
        )

        adaptive_k = np.where(spread <= 0.75, 3, 2)
        adaptive_k = np.minimum(adaptive_k, np.maximum(counts, 1))
    else:
        adaptive_k = np.ones(n, dtype=int)

    positions = np.arange(max_k)[None, :]
    selected = finite_nearest & (positions < adaptive_k[:, None])

    local_sum = np.where(selected, nearest, 0.0).sum(axis=1)
    selected_counts = selected.sum(axis=1)

    local_scale = np.divide(
        local_sum,
        np.maximum(selected_counts, 1),
        out=np.ones(n, dtype=float),
        where=selected_counts > 0,
    )
    local_scale = np.maximum(local_scale, eps)

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    normalized_distance = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    row_ratio = np.divide(
        d,
        row_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    col_ratio = np.divide(
        d,
        col_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        distance_quality = np.exp(
            -np.clip(normalized_distance, 0.0, 50.0) / 0.82
        )

        row_quality = np.exp(
            -np.clip(row_ratio, 0.0, 50.0) / 0.88
        )

        col_quality = np.exp(
            -np.clip(col_ratio, 0.0, 50.0) / 0.88
        )

        geometric_quality = np.sqrt(row_quality * col_quality)
        arithmetic_quality = 0.5 * (row_quality + col_quality)

        directional_quality = (
            0.85 * geometric_quality
            + 0.15 * arithmetic_quality
        )

        score = distance_quality * directional_quality
        score = np.where(valid, score, 0.0)

        row_max = score.max(axis=1, keepdims=True)
        exploration_floor = 0.008 * row_max
        score = np.where(valid, score + exploration_floor, 0.0)

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)

    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )
