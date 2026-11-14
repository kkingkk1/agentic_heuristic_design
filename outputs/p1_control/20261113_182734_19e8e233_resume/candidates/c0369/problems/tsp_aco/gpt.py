import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced two-sided edge priors for a directed TSP."""
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
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_min = np.min(np.where(valid, d, np.inf), axis=1)
    col_min = np.min(np.where(valid, d, np.inf), axis=0)
    row_min = np.where(row_count > 0, row_min, 0.0)
    col_min = np.where(col_count > 0, col_min, 0.0)

    row_values = np.where(valid, d, np.inf)
    row_second = np.partition(row_values, 1, axis=1)[:, 1]
    col_second = np.partition(row_values, 1, axis=0)[1, :]

    row_gap = np.where(
        row_count >= 2,
        np.maximum(row_second - row_min, 0.0),
        global_scale,
    )
    col_gap = np.where(
        col_count >= 2,
        np.maximum(col_second - col_min, 0.0),
        global_scale,
    )

    row_gap_scale = np.median(row_gap[row_gap > eps])
    col_gap_scale = np.median(col_gap[col_gap > eps])
    row_gap_scale = max(float(row_gap_scale), global_scale * 0.25, eps)
    col_gap_scale = max(float(col_gap_scale), global_scale * 0.25, eps)

    row_advantage = np.maximum(row_gap[:, None] - (d - row_min[:, None]), 0.0)
    col_advantage = np.maximum(
        col_gap[None, :] - (d - col_min[None, :]),
        0.0,
    )
    row_advantage[~valid] = 0.0
    col_advantage[~valid] = 0.0

    row_delta = np.maximum(d - row_min[:, None], 0.0)
    col_delta = np.maximum(d - col_min[None, :], 0.0)
    row_delta[~valid] = 0.0
    col_delta[~valid] = 0.0

    temperature = max(0.30 * global_scale, eps)
    affinity = np.exp(
        -np.clip(
            0.70 * row_delta / temperature
            + 0.45 * col_delta / temperature,
            0.0,
            60.0,
        )
    )
    affinity *= 1.0 + 0.85 * np.clip(
        row_advantage / row_gap_scale,
        0.0,
        1.0,
    )
    affinity *= 1.0 + 0.85 * np.clip(
        col_advantage / col_gap_scale,
        0.0,
        1.0,
    )
    affinity[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    row_sum = affinity.sum(axis=1)
    row_scale = np.divide(
        1.0,
        row_sum,
        out=np.zeros_like(row_sum),
        where=row_sum > 0.0,
    )

    col_sum = affinity.T @ row_scale
    col_scale = np.divide(
        1.0,
        col_sum,
        out=np.zeros_like(col_sum),
        where=col_sum > 0.0,
    )

    for _ in range(12):
        row_sum = affinity @ col_scale
        row_scale = np.divide(
            1.0,
            row_sum,
            out=np.zeros_like(row_sum),
            where=row_sum > 0.0,
        )

        col_sum = affinity.T @ row_scale
        col_scale = np.divide(
            1.0,
            col_sum,
            out=np.zeros_like(col_sum),
            where=col_sum > 0.0,
        )

    score = affinity * row_scale[:, None] * col_scale[None, :]
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score
