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

    eps = np.finfo(float).eps

    positive = d[valid & (d > 0.0)]
    distance_scale = (
        float(np.median(positive)) if positive.size else 1.0
    )
    distance_scale = max(distance_scale, eps)

    masked = np.where(valid, d, np.inf)

    row_candidates = np.partition(masked, kth=1, axis=1)[:, :2]
    col_candidates = np.partition(masked, kth=1, axis=0)[:2, :]

    row_min = row_candidates[:, 0]
    row_second = row_candidates[:, 1]
    col_min = col_candidates[0, :]
    col_second = col_candidates[1, :]

    valid_row_count = valid.sum(axis=1)
    valid_col_count = valid.sum(axis=0)

    row_min_safe = np.where(valid_row_count > 0, row_min, 0.0)
    col_min_safe = np.where(valid_col_count > 0, col_min, 0.0)

    row_delta = np.zeros_like(d)
    col_delta = np.zeros_like(d)

    np.subtract(
        d,
        row_min_safe[:, None],
        out=row_delta,
        where=valid,
    )
    np.maximum(row_delta, 0.0, out=row_delta, where=valid)

    np.subtract(
        d,
        col_min_safe[None, :],
        out=col_delta,
        where=valid,
    )
    np.maximum(col_delta, 0.0, out=col_delta, where=valid)

    row_gap = np.where(
        valid_row_count >= 2,
        np.maximum(row_second - row_min_safe, 0.0),
        distance_scale,
    )
    col_gap = np.where(
        valid_col_count >= 2,
        np.maximum(col_second - col_min_safe, 0.0),
        distance_scale,
    )

    positive_row_gaps = row_gap[
        (row_gap > 0.0) & np.isfinite(row_gap)
    ]
    positive_col_gaps = col_gap[
        (col_gap > 0.0) & np.isfinite(col_gap)
    ]

    row_gap_scale = (
        float(np.median(positive_row_gaps))
        if positive_row_gaps.size
        else distance_scale
    )
    col_gap_scale = (
        float(np.median(positive_col_gaps))
        if positive_col_gaps.size
        else distance_scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    regret_exponent = -0.5 * (
        row_delta / row_gap_scale
        + col_delta / col_gap_scale
    )
    regret = np.exp(np.clip(regret_exponent, -745.0, 0.0))
    regret[~valid] = 0.0

    temperature = max(distance_scale * 0.20, eps)
    logits = np.clip(row_delta / temperature, 0.0, 60.0)

    balanced = np.exp(-logits)
    balanced[~valid] = 0.0

    for _ in range(32):
        previous = balanced.copy()

        row_sum = balanced.sum(axis=1, keepdims=True)
        np.divide(
            balanced,
            row_sum,
            out=balanced,
            where=row_sum > 0.0,
        )

        col_sum = balanced.sum(axis=0, keepdims=True)
        np.divide(
            balanced,
            col_sum,
            out=balanced,
            where=col_sum > 0.0,
        )

        if np.max(np.abs(balanced - previous)) < 1e-10:
            break

    score = balanced * np.power(regret + eps, 1.1)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
