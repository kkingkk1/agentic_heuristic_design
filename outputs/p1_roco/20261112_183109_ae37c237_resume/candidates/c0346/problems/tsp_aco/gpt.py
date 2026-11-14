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

    inf = np.inf
    eps = np.finfo(float).eps
    work = np.where(valid, d, inf)

    k = min(4, n - 1)
    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    finite_nearest = np.isfinite(nearest)
    neighbor_count = finite_nearest.sum(axis=1).astype(float)

    nearest_sum = np.where(finite_nearest, nearest, 0.0).sum(axis=1)
    mean_local = np.divide(
        nearest_sum,
        np.maximum(neighbor_count, 1.0),
        out=np.ones(n, dtype=float),
        where=neighbor_count > 0.0,
    )

    best = np.min(work, axis=1)
    masked_second = np.where(work > best[:, None], work, inf)
    second = np.min(masked_second, axis=1)

    safe_best = np.where(np.isfinite(best), best, mean_local)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)

    local_scale = 0.65 * mean_local + 0.35 * safe_best
    local_scale = np.maximum(local_scale, eps)

    regret = np.clip(
        (safe_second - safe_best) / np.maximum(safe_second, eps),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.08 * regret

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    safe_distance = np.where(valid, d, 0.0)

    geometric_ratio = np.divide(
        safe_distance,
        geometric_scale,
        out=np.zeros_like(d),
        where=valid,
    )
    row_ratio = np.divide(
        safe_distance,
        row_scale,
        out=np.zeros_like(d),
        where=valid,
    )
    col_ratio = np.divide(
        safe_distance,
        col_scale,
        out=np.zeros_like(d),
        where=valid,
    )

    log_distance_quality = -np.minimum(1.20 * geometric_ratio, 40.0)
    log_mutual_quality = -0.5 * np.minimum(row_ratio + col_ratio, 80.0)

    row_rank = np.sum(
        valid & (safe_distance < safe_distance[:, :, None])
        if False
        else valid,
        axis=1,
    )

    row_ranks = np.zeros_like(d, dtype=float)
    col_ranks = np.zeros_like(d, dtype=float)

    row_ranks[valid] = (
        np.argsort(np.argsort(work, axis=1), axis=1)[valid] + 1.0
    )
    col_ranks[valid] = (
        np.argsort(np.argsort(work, axis=0), axis=0)[valid] + 1.0
    )

    log_rank_quality = -0.035 * (
        np.log1p(row_ranks) + np.log1p(col_ranks)
    )

    row_competition = np.sum(
        valid & (safe_distance <= safe_second[:, None]),
        axis=1,
        dtype=float,
    )
    col_competition = np.sum(
        valid & (safe_distance <= safe_second[None, :]),
        axis=0,
        dtype=float,
    )

    scarcity_term = 1.0 / np.sqrt(
        np.maximum(row_competition[:, None], 1.0)
        * np.maximum(col_competition[None, :], 1.0)
    )
    log_scarcity = np.log1p(0.035 * scarcity_term)

    log_urgency = (
        np.log(urgency)[:, None] + np.log(urgency)[None, :]
    )

    log_score = (
        1.00 * log_distance_quality
        + 0.55 * log_mutual_quality
        + log_rank_quality
        + log_scarcity
        + log_urgency
    )
    log_score[~valid] = -inf
    np.fill_diagonal(log_score, -inf)

    row_max = np.max(log_score, axis=1, keepdims=True)
    shifted = np.where(valid, log_score - row_max, -inf)
    score = np.exp(np.clip(shifted, -745.0, 0.0))
    score[~valid] = 0.0

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
