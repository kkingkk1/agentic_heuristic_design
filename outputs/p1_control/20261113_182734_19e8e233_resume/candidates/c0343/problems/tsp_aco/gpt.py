import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bidirectionally balanced, regret-aware TSP edge priors."""
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

    safe = np.where(valid, d, np.inf)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    row_min = np.where(row_count > 0, row_min, 0.0)
    col_min = np.where(col_count > 0, col_min, 0.0)

    row_delta = np.maximum(
        np.where(valid, d, row_min[:, None]) - row_min[:, None],
        0.0,
    )
    col_delta = np.maximum(
        np.where(valid, d, col_min[None, :]) - col_min[None, :],
        0.0,
    )

    row_second = np.full(n, scale, dtype=float)
    col_second = np.full(n, scale, dtype=float)

    row_two = row_count >= 2
    col_two = col_count >= 2

    if np.any(row_two):
        row_values = np.where(valid[row_two], d[row_two], np.inf)
        row_second[row_two] = np.partition(
            row_values,
            kth=1,
            axis=1,
        )[:, 1]

    if np.any(col_two):
        col_values = np.where(valid[:, col_two], d[:, col_two], np.inf)
        col_second[col_two] = np.partition(
            col_values,
            kth=1,
            axis=0,
        )[1, :]

    row_gap = np.maximum(row_second - row_min, 0.0)
    col_gap = np.maximum(col_second - col_min, 0.0)

    row_gap_scale = float(np.median(row_gap[row_gap > 0.0])) \
        if np.any(row_gap > 0.0) else scale
    col_gap_scale = float(np.median(col_gap[col_gap > 0.0])) \
        if np.any(col_gap > 0.0) else scale

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_regret = np.exp(
        -np.minimum(row_delta / row_gap_scale, 60.0)
    )
    col_regret = np.exp(
        -np.minimum(col_delta / col_gap_scale, 60.0)
    )

    temperature = max(0.14 * scale, eps)
    row_gibbs = np.exp(-np.minimum(row_delta / temperature, 60.0))
    col_gibbs = np.exp(-np.minimum(col_delta / temperature, 60.0))

    affinity = np.sqrt(row_gibbs * col_gibbs)
    regret = np.sqrt(row_regret * col_regret)

    affinity[~valid] = 0.0
    regret[~valid] = 0.0

    balanced = affinity.copy()

    for _ in range(80):
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

    score = balanced * np.power(regret + eps, 1.35)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
