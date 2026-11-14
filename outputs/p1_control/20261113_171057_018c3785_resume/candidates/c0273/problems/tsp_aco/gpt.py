import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return finite, regret-aware priors for directed TSP edges."""
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
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    valid_values = d[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = float(np.median(valid_values))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_best = np.where(row_count > 0, row_best, scale)
    col_best = np.where(col_count > 0, col_best, scale)

    row_second = np.full(n, scale, dtype=float)
    col_second = np.full(n, scale, dtype=float)

    rows_with_two = row_count >= 2
    cols_with_two = col_count >= 2

    if np.any(rows_with_two):
        row_second[rows_with_two] = np.partition(
            masked[rows_with_two],
            kth=1,
            axis=1,
        )[:, 1]

    if np.any(cols_with_two):
        col_second[cols_with_two] = np.partition(
            masked[:, cols_with_two],
            kth=1,
            axis=0,
        )[1, :]

    row_gap = np.maximum(row_second - row_best, 0.0)
    col_gap = np.maximum(col_second - col_best, 0.0)

    row_gap = np.where(np.isfinite(row_gap), row_gap, 0.0)
    col_gap = np.where(np.isfinite(col_gap), col_gap, 0.0)

    row_positive_gap = row_gap[row_gap > 0.0]
    col_positive_gap = col_gap[col_gap > 0.0]

    row_gap_scale = (
        float(np.median(row_positive_gap))
        if row_positive_gap.size
        else scale
    )
    col_gap_scale = (
        float(np.median(col_positive_gap))
        if col_positive_gap.size
        else scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_excess = np.zeros_like(d)
    col_excess = np.zeros_like(d)

    np.subtract(
        d,
        row_best[:, None],
        out=row_excess,
        where=valid,
    )
    np.subtract(
        d,
        col_best[None, :],
        out=col_excess,
        where=valid,
    )

    row_excess = np.maximum(row_excess, 0.0)
    col_excess = np.maximum(col_excess, 0.0)

    temperature = max(0.30 * scale, eps)

    local_logits = np.zeros_like(d)
    np.divide(
        row_excess,
        temperature,
        out=local_logits,
        where=valid,
    )
    np.clip(local_logits, 0.0, 60.0, out=local_logits)

    local_affinity = np.exp(-local_logits)
    local_affinity[~valid] = 0.0

    row_regret = np.zeros_like(d)
    col_regret = np.zeros_like(d)

    np.divide(
        row_excess,
        row_gap_scale,
        out=row_regret,
        where=valid,
    )
    np.divide(
        col_excess,
        col_gap_scale,
        out=col_regret,
        where=valid,
    )

    np.clip(row_regret, 0.0, 60.0, out=row_regret)
    np.clip(col_regret, 0.0, 60.0, out=col_regret)

    regret = np.exp(-0.5 * (row_regret + col_regret))
    regret[~valid] = 0.0

    balanced = local_affinity.copy()

    for _ in range(72):
        previous = balanced.copy()

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

        if np.max(np.abs(balanced - previous)) < 1e-10:
            break

    combined = 0.50 * balanced + 0.50 * local_affinity
    score = combined * np.power(regret + eps, 0.9)

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
