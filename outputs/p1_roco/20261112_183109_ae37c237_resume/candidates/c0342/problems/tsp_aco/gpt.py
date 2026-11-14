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

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    k = min(4, n - 1)
    local_neighbors = np.partition(work, kth=k - 1, axis=1)[:, :k]
    finite_neighbors = np.isfinite(local_neighbors)

    counts = finite_neighbors.sum(axis=1)
    neighbor_sum = np.where(finite_neighbors, local_neighbors, 0.0).sum(axis=1)

    local_scale = np.divide(
        neighbor_sum,
        np.maximum(counts, 1),
        out=np.ones(n, dtype=float),
        where=counts > 0,
    )
    local_scale = np.maximum(local_scale, eps)

    sorted_rows = np.sort(local_neighbors, axis=1)
    best = sorted_rows[:, 0]
    second = sorted_rows[:, 1] if k > 1 else best.copy()

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)

    regret = np.clip(
        (safe_second - safe_best) / np.maximum(safe_second, eps),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.16 * regret

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    normalized = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    geometric_quality = np.exp(
        -np.clip(1.18 * normalized, 0.0, 50.0)
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

    row_quality = 1.0 / (1.0 + np.clip(row_ratio, 0.0, 50.0))
    col_quality = 1.0 / (1.0 + np.clip(col_ratio, 0.0, 50.0))
    mutual_quality = np.sqrt(row_quality * col_quality)

    local_competition = np.sum(
        valid & (d <= safe_second[:, None]),
        axis=1,
        dtype=float,
    )
    reverse_competition = np.sum(
        valid & (d <= safe_second[None, :]),
        axis=0,
        dtype=float,
    )

    scarcity = 1.0 + 0.06 / np.sqrt(
        np.maximum(local_competition[:, None], 1.0)
        * np.maximum(reverse_competition[None, :], 1.0)
    )

    score = (
        geometric_quality
        * mutual_quality
        * scarcity
        * urgency[:, None]
        * urgency[None, :]
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)

    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
