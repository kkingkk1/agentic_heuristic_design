import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced, regret-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    eps = np.finfo(float).eps

    positive = d[valid & (d > 0.0)]
    if positive.size:
        scale = float(np.median(positive))
    else:
        finite = d[valid]
        scale = float(np.median(finite)) if finite.size else 1.0
    scale = max(scale, eps)

    safe = np.where(valid, d, np.inf)

    row_best = np.min(safe, axis=1)
    col_best = np.min(safe, axis=0)

    row_has_edge = np.isfinite(row_best)
    col_has_edge = np.isfinite(col_best)

    row_base = np.where(row_has_edge, row_best, 0.0)
    col_base = np.where(col_has_edge, col_best, 0.0)

    row_delta = np.where(
        valid,
        np.maximum(d - row_base[:, None], 0.0),
        0.0,
    )
    col_delta = np.where(
        valid,
        np.maximum(d - col_base[None, :], 0.0),
        0.0,
    )

    row_second = np.partition(safe, kth=1, axis=1)[:, 1]
    col_second = np.partition(safe, kth=1, axis=0)[1, :]

    row_gap = np.where(
        row_has_edge & np.isfinite(row_second),
        np.maximum(row_second - row_best, 0.0),
        0.0,
    )
    col_gap = np.where(
        col_has_edge & np.isfinite(col_second),
        np.maximum(col_second - col_best, 0.0),
        0.0,
    )

    row_gap_values = row_gap[row_gap > 0.0]
    col_gap_values = col_gap[col_gap > 0.0]

    row_gap_scale = (
        float(np.median(row_gap_values))
        if row_gap_values.size
        else scale
    )
    col_gap_scale = (
        float(np.median(col_gap_values))
        if col_gap_values.size
        else scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    temperature = max(0.26 * scale, eps)

    row_affinity = np.exp(-np.minimum(row_delta / temperature, 60.0))
    col_affinity = np.exp(-np.minimum(col_delta / temperature, 60.0))

    affinity = np.sqrt(row_affinity * col_affinity)
    affinity[~valid] = 0.0

    row_regret = np.exp(-np.minimum(row_delta / row_gap_scale, 60.0))
    col_regret = np.exp(-np.minimum(col_delta / col_gap_scale, 60.0))

    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    balanced = affinity.copy()

    for _ in range(48):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_sum,
            out=np.zeros_like(balanced),
            where=row_sum > 0.0,
        )

        col_sum = balanced.sum(axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            col_sum,
            out=np.zeros_like(balanced),
            where=col_sum > 0.0,
        )

        balanced[~valid] = 0.0

    score = balanced * (0.30 + 0.70 * np.power(regret, 0.85))
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
