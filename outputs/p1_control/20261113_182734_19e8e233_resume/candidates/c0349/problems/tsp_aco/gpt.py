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

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_best = np.where(row_count > 0, row_best, 0.0)
    col_best = np.where(col_count > 0, col_best, 0.0)

    row_delta = np.maximum(d - row_best[:, None], 0.0)
    col_delta = np.maximum(d - col_best[None, :], 0.0)
    row_delta[~valid] = 0.0
    col_delta[~valid] = 0.0

    row_second = np.full(n, np.inf, dtype=float)
    col_second = np.full(n, np.inf, dtype=float)

    rows_with_two = row_count >= 2
    cols_with_two = col_count >= 2

    if np.any(rows_with_two):
        row_values = np.where(valid[rows_with_two], d[rows_with_two], np.inf)
        row_second[rows_with_two] = np.partition(
            row_values, 1, axis=1
        )[:, 1]

    if np.any(cols_with_two):
        col_values = np.where(valid[:, cols_with_two], d[:, cols_with_two], np.inf)
        col_second[cols_with_two] = np.partition(
            col_values, 1, axis=0
        )[1, :]

    row_gap = np.where(
        rows_with_two,
        np.maximum(row_second - row_best, 0.0),
        scale,
    )
    col_gap = np.where(
        cols_with_two,
        np.maximum(col_second - col_best, 0.0),
        scale,
    )

    row_gap_values = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_gap_values = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

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

    regret = np.exp(
        -0.5
        * (
            row_delta / row_gap_scale
            + col_delta / col_gap_scale
        )
    )
    regret[~valid] = 0.0

    temperature = max(0.12 * scale, eps)
    logits = 0.5 * (row_delta + col_delta) / temperature
    logits = np.clip(logits, 0.0, 60.0)

    affinity = np.exp(-logits)
    affinity[~valid] = 0.0

    balanced = affinity.copy()

    for _ in range(48):
        previous = balanced

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

        change = np.max(np.abs(balanced - previous))
        if change < 1e-10:
            break

    score = balanced * np.power(regret + eps, 1.25)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
