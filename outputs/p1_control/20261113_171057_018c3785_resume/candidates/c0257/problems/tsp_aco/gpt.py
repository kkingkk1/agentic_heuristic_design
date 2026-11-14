import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return lightweight bidirectionally normalized priors for directed TSP edges."""
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

    values = d[valid]
    positive = values[values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    masked = np.where(valid, d, np.inf)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_second = np.partition(masked, min(1, n - 1), axis=1)[:, 1]
    col_second = np.partition(masked, min(1, n - 1), axis=0)[1, :]

    row_gap = row_second - row_best
    col_gap = col_second - col_best

    row_gap = np.where(
        np.isfinite(row_gap) & (row_gap > 0.0),
        row_gap,
        scale,
    )
    col_gap = np.where(
        np.isfinite(col_gap) & (col_gap > 0.0),
        col_gap,
        scale,
    )

    row_excess = np.maximum(d - row_best[:, None], 0.0)
    col_excess = np.maximum(d - col_best[None, :], 0.0)

    temperature = max(0.5 * scale, np.finfo(float).eps)

    with np.errstate(over="ignore", invalid="ignore"):
        local_affinity = np.exp(
            -np.minimum(
                0.65 * row_excess / temperature
                + 0.35 * col_excess / temperature,
                55.0,
            )
        )

        regret = np.exp(
            -np.minimum(
                0.5 * row_excess / row_gap[:, None]
                + 0.5 * col_excess / col_gap[None, :],
                55.0,
            )
        )

    score = local_affinity * np.power(regret, 0.8)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    row_max = np.max(score, axis=1, keepdims=True)
    col_max = np.max(score, axis=0, keepdims=True)

    row_max[row_max <= 0.0] = 1.0
    col_max[col_max <= 0.0] = 1.0

    score = score * np.sqrt(
        np.divide(score, row_max, out=np.zeros_like(score), where=row_max > 0.0)
        + np.divide(score, col_max, out=np.zeros_like(score), where=col_max > 0.0)
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
