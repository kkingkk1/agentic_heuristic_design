import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return fast, feasibility-preserving edge priors for directed TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    base_scale = float(np.median(positive)) if positive.size else 1.0
    tiny = np.finfo(float).eps
    base_scale = max(base_scale, tiny)

    safe = np.where(valid, d, np.inf)

    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    row_second = np.partition(safe, 1, axis=1)[:, 1]
    col_second = np.partition(safe, 1, axis=0)[1, :]

    row_gap = np.maximum(row_second - row_min, 0.0)
    col_gap = np.maximum(col_second - col_min, 0.0)

    row_gaps = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_gaps = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_scale = float(np.median(row_gaps)) if row_gaps.size else base_scale
    col_scale = float(np.median(col_gaps)) if col_gaps.size else base_scale
    row_scale = max(row_scale, tiny)
    col_scale = max(col_scale, tiny)

    row_delta = np.where(
        valid,
        np.maximum(d - row_min[:, None], 0.0),
        0.0,
    )
    col_delta = np.where(
        valid,
        np.maximum(d - col_min[None, :], 0.0),
        0.0,
    )

    row_affinity = np.exp(-np.clip(row_delta / row_scale, 0.0, 60.0))
    col_affinity = np.exp(-np.clip(col_delta / col_scale, 0.0, 60.0))

    affinity = np.sqrt(row_affinity * col_affinity)
    regret = np.sqrt(row_affinity * col_affinity)

    score = affinity * np.power(regret + tiny, 0.7)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
