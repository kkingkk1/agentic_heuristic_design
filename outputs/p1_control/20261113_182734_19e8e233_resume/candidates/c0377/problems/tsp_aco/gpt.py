import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diverse, bidirectional local priors for directed TSP edges."""
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

    safe = np.where(valid, d, 0.0)
    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_min = np.min(np.where(valid, d, np.inf), axis=1)
    col_min = np.min(np.where(valid, d, np.inf), axis=0)

    row_min = np.where(row_count > 0, row_min, 0.0)
    col_min = np.where(col_count > 0, col_min, 0.0)

    row_delta = np.where(
        valid,
        np.maximum(safe - row_min[:, None], 0.0),
        0.0,
    )
    col_delta = np.where(
        valid,
        np.maximum(safe - col_min[None, :], 0.0),
        0.0,
    )

    row_second = np.full(n, np.inf, dtype=float)
    col_second = np.full(n, np.inf, dtype=float)

    rows_two = row_count >= 2
    cols_two = col_count >= 2

    if np.any(rows_two):
        masked_rows = np.where(valid[rows_two], d[rows_two], np.inf)
        row_second[rows_two] = np.partition(
            masked_rows,
            kth=1,
            axis=1,
        )[:, 1]

    if np.any(cols_two):
        masked_cols = np.where(valid[:, cols_two], d[:, cols_two], np.inf)
        col_second[cols_two] = np.partition(
            masked_cols,
            kth=1,
            axis=0,
        )[1, :]

    row_gap = np.where(
        rows_two,
        np.maximum(row_second - row_min, 0.0),
        scale,
    )
    col_gap = np.where(
        cols_two,
        np.maximum(col_second - col_min, 0.0),
        scale,
    )

    row_gap_values = row_gap[
        (row_gap > 0.0) & np.isfinite(row_gap)
    ]
    col_gap_values = col_gap[
        (col_gap > 0.0) & np.isfinite(col_gap)
    ]

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

    temperature = max(0.24 * scale, eps)
    geometric = np.exp(
        -0.5
        * (
            row_delta / temperature
            + col_delta / temperature
        )
    )
    geometric[~valid] = 0.0

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[np.argsort(d[i, indices], kind="stable")]
            denominator = max(order.size - 1, 1)
            row_rank[i, order] = (
                1.0 - np.arange(order.size, dtype=float) / denominator
            )

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = indices[
                np.argsort(d[indices, j], kind="stable")
            ]
            denominator = max(order.size - 1, 1)
            col_rank[order, j] = (
                1.0 - np.arange(order.size, dtype=float) / denominator
            )

    mutual_nearest = (
        valid
        & (row_rank >= 1.0)
        & (col_rank >= 1.0)
    )

    rank_bonus = (
        1.0
        + 0.16 * row_rank
        + 0.16 * col_rank
        + 0.14 * mutual_nearest.astype(float)
    )
    rank_bonus[~valid] = 0.0

    affinity = geometric * rank_bonus
    affinity *= np.power(regret + eps, 0.65)
    affinity[~valid] = 0.0

    balanced = affinity.copy()

    for _ in range(32):
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

    valid_float = valid.astype(float)
    row_valid = valid_float / np.maximum(
        valid_float.sum(axis=1, keepdims=True),
        1.0,
    )
    diversity_floor = 0.035 * row_valid

    score = 0.965 * balanced + diversity_floor
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
