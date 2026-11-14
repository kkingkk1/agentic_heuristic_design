import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced, regret-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    eps = np.finfo(float).eps
    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        distance_scale = float(np.median(positive_values))
    else:
        distance_scale = max(float(np.median(finite_values)), 1.0)
    distance_scale = max(distance_scale, eps)

    masked = np.where(valid, d, np.inf)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_second = np.partition(masked, 1, axis=1)[:, 1]
    col_second = np.partition(masked, 1, axis=0)[1, :]

    row_gap = row_second - row_best
    col_gap = col_second - col_best

    row_positive_gaps = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_positive_gaps = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(row_positive_gaps))
        if row_positive_gaps.size
        else distance_scale
    )
    col_gap_scale = (
        float(np.median(col_positive_gaps))
        if col_positive_gaps.size
        else distance_scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_excess = np.zeros_like(d)
    col_excess = np.zeros_like(d)

    np.subtract(d, row_best[:, None], out=row_excess, where=valid)
    np.subtract(d, col_best[None, :], out=col_excess, where=valid)

    row_excess = np.maximum(row_excess, 0.0)
    col_excess = np.maximum(col_excess, 0.0)

    row_relative = row_excess / row_gap_scale
    col_relative = col_excess / col_gap_scale

    combined_relative = 0.5 * (row_relative + col_relative)

    temperature = max(0.22 * distance_scale, eps)
    distance_cost = np.minimum(
        combined_relative + 0.5 * (row_excess + col_excess) / temperature,
        60.0,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        log_affinity = -distance_cost
        log_affinity[~valid] = -np.inf

    def logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
        maximum = np.max(values, axis=axis, keepdims=True)
        finite_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
        shifted = values - finite_maximum
        shifted = np.where(np.isfinite(shifted), shifted, -np.inf)
        summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
        result = finite_maximum + np.log(np.maximum(summed, eps))
        return np.squeeze(result, axis=axis)

    row_potential = np.zeros(n, dtype=float)
    col_potential = np.zeros(n, dtype=float)

    for _ in range(32):
        row_potential = -logsumexp(
            log_affinity + col_potential[None, :],
            axis=1,
        )
        row_potential[~np.isfinite(row_potential)] = 0.0

        col_potential = -logsumexp(
            log_affinity + row_potential[:, None],
            axis=0,
        )
        col_potential[~np.isfinite(col_potential)] = 0.0

    balanced_log_affinity = (
        log_affinity
        + row_potential[:, None]
        + col_potential[None, :]
    )

    row_regret = np.exp(-np.minimum(row_relative, 60.0))
    col_regret = np.exp(-np.minimum(col_relative, 60.0))
    regret = np.sqrt(row_regret * col_regret)

    score = np.exp(np.clip(balanced_log_affinity, -745.0, 0.0))
    score *= np.power(regret + eps, 0.8)

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
