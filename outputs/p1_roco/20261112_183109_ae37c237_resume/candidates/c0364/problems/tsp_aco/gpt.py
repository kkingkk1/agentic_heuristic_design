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

    k = min(4, n - 1)
    nearest = sorted_rows[:, :k]
    finite_nearest = np.isfinite(nearest)
    counts = finite_nearest.sum(axis=1)

    local_sum = np.where(finite_nearest, nearest, 0.0).sum(axis=1)
    local_scale = np.divide(
        local_sum,
        np.maximum(counts, 1),
        out=np.ones(n, dtype=float),
        where=counts > 0,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    best = sorted_rows[:, 0]
    if n > 2:
        second = sorted_rows[:, 1]
    else:
        second = best.copy()

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(
        np.isfinite(second),
        second,
        2.0 * safe_best,
    )

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, np.finfo(float).eps),
        0.0,
        1.0,
    )

    # Mildly prioritize rows whose best edge is substantially better
    # than their alternatives, without dominating distance quality.
    urgency = 1.0 + 0.08 * regret

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
            -np.clip(normalized_distance, 0.0, 50.0) / 0.70
        )

        row_quality = np.exp(
            -np.clip(row_ratio, 0.0, 50.0) / 0.80
        )
        col_quality = np.exp(
            -np.clip(col_ratio, 0.0, 50.0) / 0.80
        )

        mutual_quality = np.sqrt(row_quality * col_quality)

        score = (
            distance_quality
            * mutual_quality
            * urgency[:, None]
        )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)

    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )
