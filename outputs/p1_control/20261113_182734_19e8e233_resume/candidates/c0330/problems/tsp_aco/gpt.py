import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, regret-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    result = np.zeros((n, n), dtype=float)
    if not np.any(valid):
        return result

    positive = d[valid & (d > 0.0)]
    if positive.size:
        scale = float(np.median(positive))
    else:
        finite = d[valid]
        scale = float(np.median(finite)) if finite.size else 1.0
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)
    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_min = np.min(masked, axis=1)
    col_min = np.min(masked, axis=0)

    row_min = np.where(row_count > 0, row_min, 0.0)
    col_min = np.where(col_count > 0, col_min, 0.0)

    row_delta = np.zeros_like(d)
    col_delta = np.zeros_like(d)

    np.subtract(d, row_min[:, None], out=row_delta, where=valid)
    np.subtract(d, col_min[None, :], out=col_delta, where=valid)
    np.maximum(row_delta, 0.0, out=row_delta, where=valid)
    np.maximum(col_delta, 0.0, out=col_delta, where=valid)

    row_second = np.full(n, np.inf, dtype=float)
    col_second = np.full(n, np.inf, dtype=float)

    rows_with_two = row_count >= 2
    cols_with_two = col_count >= 2

    if np.any(rows_with_two):
        row_values = masked[rows_with_two]
        row_second[rows_with_two] = np.partition(
            row_values, kth=1, axis=1
        )[:, 1]

    if np.any(cols_with_two):
        col_values = masked[:, cols_with_two]
        col_second[cols_with_two] = np.partition(
            col_values, kth=1, axis=0
        )[1, :]

    row_gap = np.where(
        rows_with_two,
        np.maximum(row_second - row_min, 0.0),
        scale,
    )
    col_gap = np.where(
        cols_with_two,
        np.maximum(col_second - col_min, 0.0),
        scale,
    )

    row_positive = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_positive = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(row_positive)) if row_positive.size else scale
    )
    col_gap_scale = (
        float(np.median(col_positive)) if col_positive.size else scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    with np.errstate(over="ignore", invalid="ignore"):
        regret_log = -0.5 * (
            row_delta / row_gap_scale
            + col_delta / col_gap_scale
        )
        regret = np.exp(np.clip(regret_log, -60.0, 0.0))

    regret[~valid] = 0.0

    temperature = max(0.18 * scale, eps)
    combined_delta = 0.5 * (row_delta + col_delta)

    with np.errstate(over="ignore", invalid="ignore"):
        logits = np.clip(combined_delta / temperature, 0.0, 60.0)
        affinity = np.exp(-logits)

    affinity[~valid] = 0.0

    balanced = affinity.copy()
    max_iter = 64
    tolerance = 1e-10

    for _ in range(max_iter):
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

        change = float(np.max(np.abs(balanced - previous)))
        if change <= tolerance:
            break

    score = balanced * np.power(regret + eps, 0.9)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
