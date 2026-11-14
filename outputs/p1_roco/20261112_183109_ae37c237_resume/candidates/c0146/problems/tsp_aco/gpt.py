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
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    work = np.where(valid, d, np.inf)
    eps = np.finfo(float).eps

    row_sorted = np.sort(work, axis=1)
    col_sorted = np.sort(work, axis=0)

    row_best = row_sorted[:, 0]
    col_best = col_sorted[0, :]

    row_second = row_sorted[:, 1] if n > 2 else row_best.copy()
    col_second = col_sorted[1, :] if n > 2 else col_best.copy()

    row_best_safe = np.where(np.isfinite(row_best), row_best, 1.0)
    col_best_safe = np.where(np.isfinite(col_best), col_best, 1.0)

    row_second_safe = np.where(
        np.isfinite(row_second),
        row_second,
        2.0 * row_best_safe,
    )
    col_second_safe = np.where(
        np.isfinite(col_second),
        col_second,
        2.0 * col_best_safe,
    )

    row_scale = np.maximum(
        0.5 * (row_best_safe + row_second_safe),
        eps,
    )
    col_scale = np.maximum(
        0.5 * (col_best_safe + col_second_safe),
        eps,
    )

    row_regret = np.clip(
        (row_second_safe - row_best_safe)
        / np.maximum(row_second_safe, eps),
        0.0,
        1.0,
    )
    col_regret = np.clip(
        (col_second_safe - col_best_safe)
        / np.maximum(col_second_safe, eps),
        0.0,
        1.0,
    )

    source_urgency = 1.0 + 0.6 * row_regret
    destination_urgency = 1.0 + 0.6 * col_regret

    row_ratio = np.divide(
        d,
        row_scale[:, None],
        out=np.full_like(d, np.inf),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        col_scale[None, :],
        out=np.full_like(d, np.inf),
        where=valid,
    )

    row_quality = np.exp(-np.clip(row_ratio, 0.0, 60.0))
    col_quality = np.exp(-np.clip(col_ratio, 0.0, 60.0))
    mutual_quality = np.sqrt(row_quality * col_quality)

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        order = indices[np.argsort(d[i, indices], kind="stable")]
        row_rank[i, order] = np.arange(order.size, dtype=float)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        order = indices[np.argsort(d[indices, j], kind="stable")]
        col_rank[order, j] = np.arange(order.size, dtype=float)

    rank_quality = 1.0 / (1.0 + row_rank + col_rank)

    destination_degree = valid.sum(axis=0).astype(float)
    destination_degree = np.maximum(destination_degree, 1.0)
    destination_diversity = 1.0 / np.sqrt(destination_degree)
    destination_diversity /= np.max(destination_diversity)

    score = (
        mutual_quality
        * rank_quality
        * source_urgency[:, None]
        * destination_urgency[None, :]
        * (0.9 + 0.1 * destination_diversity[None, :])
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
