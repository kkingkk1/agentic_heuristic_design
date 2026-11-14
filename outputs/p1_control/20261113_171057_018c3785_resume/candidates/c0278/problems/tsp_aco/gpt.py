import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, regret-aware edge priors for directed TSP."""
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
    values = d[valid]
    positive = values[values > 0.0]

    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)

    row_min = np.min(masked, axis=1)
    col_min = np.min(masked, axis=0)

    row_second = np.partition(masked, 1, axis=1)[:, 1]
    col_second = np.partition(masked, 1, axis=0)[1, :]

    row_gap = row_second - row_min
    col_gap = col_second - col_min

    row_positive = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_positive = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(row_positive)) if row_positive.size else scale
    )
    col_gap_scale = (
        float(np.median(col_positive)) if col_positive.size else scale
    )

    if not np.isfinite(row_gap_scale) or row_gap_scale <= 0.0:
        row_gap_scale = scale
    if not np.isfinite(col_gap_scale) or col_gap_scale <= 0.0:
        col_gap_scale = scale

    row_delta = np.zeros_like(d)
    col_delta = np.zeros_like(d)

    with np.errstate(invalid="ignore", over="ignore"):
        np.subtract(
            d,
            row_min[:, None],
            out=row_delta,
            where=valid,
        )
        np.subtract(
            d,
            col_min[None, :],
            out=col_delta,
            where=valid,
        )

    np.maximum(row_delta, 0.0, out=row_delta)
    np.maximum(col_delta, 0.0, out=col_delta)

    temperature = max(0.42 * scale, eps)

    row_logits = np.zeros_like(d)
    col_logits = np.zeros_like(d)

    np.divide(row_delta, temperature, out=row_logits, where=valid)
    np.divide(col_delta, temperature, out=col_logits, where=valid)

    np.clip(row_logits, 0.0, 60.0, out=row_logits)
    np.clip(col_logits, 0.0, 60.0, out=col_logits)

    affinity = np.exp(-0.5 * (row_logits + col_logits))
    affinity[~valid] = 0.0
    affinity[~np.isfinite(affinity)] = 0.0

    row_regret = np.zeros_like(d)
    col_regret = np.zeros_like(d)

    np.divide(row_delta, row_gap_scale, out=row_regret, where=valid)
    np.divide(col_delta, col_gap_scale, out=col_regret, where=valid)

    np.clip(row_regret, 0.0, 60.0, out=row_regret)
    np.clip(col_regret, 0.0, 60.0, out=col_regret)

    regret = np.exp(-0.5 * (row_regret + col_regret))
    regret[~valid] = 0.0
    regret[~np.isfinite(regret)] = 0.0

    balanced = affinity.copy()

    for _ in range(8):
        row_sum = np.sum(balanced, axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_sum,
            out=np.zeros_like(balanced),
            where=np.isfinite(row_sum) & (row_sum > 0.0),
        )

        col_sum = np.sum(balanced, axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            col_sum,
            out=np.zeros_like(balanced),
            where=np.isfinite(col_sum) & (col_sum > 0.0),
        )

        balanced[~np.isfinite(balanced)] = 0.0

    combined = 0.55 * balanced + 0.45 * affinity
    score = combined * np.power(regret + eps, 0.8)

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
