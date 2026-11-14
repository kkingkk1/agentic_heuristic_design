import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return low-cost, mutually balanced priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    values = d[valid]
    positive = values[values > 0.0]

    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)

    row_best = masked.min(axis=1)
    col_best = masked.min(axis=0)

    row_second = np.partition(masked, 1, axis=1)[:, 1]
    col_second = np.partition(masked, 1, axis=0)[1, :]

    row_gap = row_second - row_best
    col_gap = col_second - col_best

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

    row_excess = np.zeros_like(d)
    col_excess = np.zeros_like(d)

    np.subtract(d, row_best[:, None], out=row_excess, where=valid)
    np.subtract(d, col_best[None, :], out=col_excess, where=valid)

    temperature = max(0.42 * scale, eps)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        local = np.zeros_like(d)
        np.divide(row_excess, temperature, out=local, where=valid)
        np.minimum(local, 50.0, out=local)
        np.negative(local, out=local)
        np.exp(local, out=local)

        row_regret = np.zeros_like(d)
        np.divide(row_excess, row_gap_scale, out=row_regret, where=valid)
        np.minimum(row_regret, 50.0, out=row_regret)
        np.negative(row_regret, out=row_regret)
        np.exp(row_regret, out=row_regret)

        col_regret = np.zeros_like(d)
        np.divide(col_excess, col_gap_scale, out=col_regret, where=valid)
        np.minimum(col_regret, 50.0, out=col_regret)
        np.negative(col_regret, out=col_regret)
        np.exp(col_regret, out=col_regret)

        score = local * np.sqrt(row_regret * col_regret)

    score[~valid] = 0.0

    row_max = score.max(axis=1, keepdims=True)
    row_max[row_max <= 0.0] = 1.0
    score /= row_max

    col_max = score.max(axis=0, keepdims=True)
    col_max[col_max <= 0.0] = 1.0
    score *= np.sqrt(score / col_max + eps)

    row_sum = score.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    score /= row_sum

    maximum = float(score.max())
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
