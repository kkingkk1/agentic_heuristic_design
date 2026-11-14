import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return finite, regret-aware edge priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    tiny = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]
    base_scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(base_scale) or base_scale <= 0.0:
        base_scale = 1.0
    base_scale = max(base_scale, tiny)

    safe = np.where(valid, d, np.inf)

    row_has = valid.any(axis=1)
    col_has = valid.any(axis=0)

    row_min = np.full(n, base_scale, dtype=float)
    col_min = np.full(n, base_scale, dtype=float)
    row_min[row_has] = np.min(safe[row_has], axis=1)
    col_min[col_has] = np.min(safe[:, col_has], axis=0)

    row_sorted = np.sort(safe, axis=1)
    col_sorted = np.sort(safe, axis=0)

    row_second = np.full(n, base_scale, dtype=float)
    col_second = np.full(n, base_scale, dtype=float)
    row_second[row_has] = row_sorted[row_has, 1]
    col_second[col_has] = col_sorted[1, col_has]

    row_gap = np.maximum(row_second - row_min, 0.0)
    col_gap = np.maximum(col_second - col_min, 0.0)
    row_gap[~np.isfinite(row_gap)] = 0.0
    col_gap[~np.isfinite(col_gap)] = 0.0

    row_gaps = row_gap[row_gap > 0.0]
    col_gaps = col_gap[col_gap > 0.0]

    row_scale = float(np.median(row_gaps)) if row_gaps.size else base_scale
    col_scale = float(np.median(col_gaps)) if col_gaps.size else base_scale

    if not np.isfinite(row_scale) or row_scale <= 0.0:
        row_scale = base_scale
    if not np.isfinite(col_scale) or col_scale <= 0.0:
        col_scale = base_scale

    row_delta = np.zeros((n, n), dtype=float)
    col_delta = np.zeros((n, n), dtype=float)

    row_delta[valid] = np.maximum(
        d[valid] - np.broadcast_to(row_min[:, None], d.shape)[valid],
        0.0,
    )
    col_delta[valid] = np.maximum(
        d[valid] - np.broadcast_to(col_min[None, :], d.shape)[valid],
        0.0,
    )

    row_affinity = np.exp(
        -np.clip(row_delta / max(row_scale, tiny), 0.0, 60.0)
    )
    col_affinity = np.exp(
        -np.clip(col_delta / max(col_scale, tiny), 0.0, 60.0)
    )

    row_affinity[~valid] = 0.0
    col_affinity[~valid] = 0.0

    two_sided_affinity = np.sqrt(row_affinity * col_affinity)
    combined_affinity = (
        0.55 * row_affinity + 0.45 * two_sided_affinity
    )

    regret = two_sided_affinity
    score = combined_affinity * np.power(regret + tiny, 0.8)

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
