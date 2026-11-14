import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed TSP edge priors from ranks, detour savings, and reciprocity."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    eps = np.finfo(float).eps
    safe = np.where(valid, d, np.inf)

    finite = d[valid]
    positive = finite[finite > 0.0]
    scale = float(np.median(positive)) if positive.size else float(np.median(finite))
    scale = max(scale, eps)

    row_order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)
    row_rank = np.where(valid, row_rank, np.inf)

    col_order = np.argsort(safe, axis=0, kind="stable")
    col_rank = np.empty_like(col_order, dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n, dtype=float)
    col_rank = np.where(valid, col_rank, np.inf)

    valid_row_counts = np.sum(valid, axis=1, keepdims=True).astype(float)
    valid_col_counts = np.sum(valid, axis=0, keepdims=True).astype(float)

    row_rank_score = np.divide(
        1.0,
        1.0 + row_rank,
        out=np.zeros((n, n), dtype=float),
        where=np.isfinite(row_rank),
    )
    col_rank_score = np.divide(
        1.0,
        1.0 + col_rank,
        out=np.zeros((n, n), dtype=float),
        where=np.isfinite(col_rank),
    )

    row_best = np.min(safe, axis=1)
    col_best = np.min(safe, axis=0)

    row_gap = np.partition(safe, 1, axis=1)[:, 1] - row_best
    col_gap = np.partition(safe, 1, axis=0)[1, :] - col_best

    row_gap = np.where(np.isfinite(row_gap) & (row_gap > 0.0), row_gap, scale)
    col_gap = np.where(np.isfinite(col_gap) & (col_gap > 0.0), col_gap, scale)

    row_contrast = np.divide(
        np.maximum(d - row_best[:, None], 0.0),
        row_gap[:, None],
        out=np.zeros_like(d),
        where=valid,
    )
    col_contrast = np.divide(
        np.maximum(d - col_best[None, :], 0.0),
        col_gap[None, :],
        out=np.zeros_like(d),
        where=valid,
    )

    contrast_score = 1.0 / (
        1.0 + np.sqrt(np.maximum(row_contrast, 0.0) *
                      np.maximum(col_contrast, 0.0))
    )

    incoming = safe.T
    outgoing = safe

    with np.errstate(invalid="ignore", over="ignore"):
        two_step = np.min(
            outgoing[:, :, None] + incoming[None, :, :],
            axis=1,
        )

    detour_gap = np.maximum(two_step - d, 0.0)
    detour_scale = np.median(detour_gap[valid]) if np.any(detour_gap[valid] > 0.0) else scale
    detour_scale = max(float(detour_scale), eps)

    detour_score = np.divide(
        detour_gap,
        detour_gap + detour_scale,
        out=np.zeros_like(d),
        where=valid,
    )

    reverse_valid = valid.T
    reverse_distance = np.where(reverse_valid, d.T, np.inf)
    reciprocal = np.divide(
        1.0,
        1.0 + np.maximum(d - reverse_distance, 0.0) / scale,
        out=np.zeros_like(d),
        where=valid & reverse_valid,
    )
    reciprocal = np.where(valid & reverse_valid, reciprocal, 0.5)

    rank_score = np.sqrt(row_rank_score * col_rank_score)

    score = (
        0.42 * rank_score
        + 0.28 * contrast_score
        + 0.20 * detour_score
        + 0.10 * reciprocal
    )

    score[~valid] = 0.0

    row_mass = score.sum(axis=1, keepdims=True)
    row_mass = np.maximum(row_mass, eps)
    score = np.divide(
        score,
        row_mass,
        out=np.zeros_like(score),
        where=row_mass > 0.0,
    )

    score = np.power(score, 0.72)
    score *= np.sqrt(np.maximum(valid_row_counts, 1.0) / np.maximum(valid_col_counts, 1.0))

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
