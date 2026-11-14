import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0)

    if np.any(valid):
        valid_values = d[valid]
        fallback = float(np.percentile(valid_values, 90))
        if not np.isfinite(fallback) or fallback <= 0:
            fallback = float(np.max(valid_values))
    else:
        return np.zeros_like(d, dtype=float)

    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    clean = np.where(valid, d, fallback)
    np.fill_diagonal(clean, np.inf)

    finite_offdiag = clean[np.isfinite(clean)]
    if finite_offdiag.size == 0:
        return np.zeros_like(d, dtype=float)

    positive = finite_offdiag[finite_offdiag > 0]
    positive_scale = (
        float(np.median(positive)) if positive.size else 1.0
    )
    if not np.isfinite(positive_scale) or positive_scale <= 0:
        positive_scale = 1.0

    safe_distance = np.where(
        np.isfinite(clean),
        np.maximum(clean, 1e-12),
        np.inf,
    )

    inverse = np.where(
        np.isfinite(safe_distance),
        positive_scale / safe_distance,
        0.0,
    )

    order = np.argsort(safe_distance, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_positions = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_positions

    valid_mask = np.isfinite(safe_distance)
    neighbor_count = np.maximum(valid_mask.sum(axis=1, keepdims=True) - 1, 1)
    rank_fraction = ranks / neighbor_count
    rank_score = np.exp(-3.0 * rank_fraction)
    rank_score[~valid_mask] = 0.0

    row_median = np.median(
        np.where(valid_mask, safe_distance, np.nan),
        axis=1,
    )
    row_median = np.where(
        np.isfinite(row_median) & (row_median > 0),
        row_median,
        positive_scale,
    )

    contrast = np.where(
        valid_mask,
        np.exp(-safe_distance / row_median[:, None]),
        0.0,
    )

    row_max = np.max(inverse, axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    inverse = inverse / row_max

    score = (
        0.55 * inverse
        + 0.30 * rank_score
        + 0.15 * contrast
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.clip(score, 0.0, None)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite_scores = score[off_diagonal & np.isfinite(score)]

    if finite_scores.size:
        upper = float(np.percentile(finite_scores, 99))
        if np.isfinite(upper) and upper > 0:
            score = np.clip(score / upper, 0.0, 1.0)

    np.fill_diagonal(score, 0.0)
    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
