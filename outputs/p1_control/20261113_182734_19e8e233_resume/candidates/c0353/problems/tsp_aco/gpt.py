import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced, regret-aware priors for directed TSP edges."""
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
    distance_scale = (
        float(np.median(positive)) if positive.size else 1.0
    )
    distance_scale = max(distance_scale, eps)

    safe = np.where(valid, d, 0.0)
    valid_count = valid.sum(axis=1)
    valid_col_count = valid.sum(axis=0)

    row_min = np.min(
        np.where(valid, d, np.inf),
        axis=1,
    )
    col_min = np.min(
        np.where(valid, d, np.inf),
        axis=0,
    )

    row_min_safe = np.where(valid_count > 0, row_min, 0.0)
    col_min_safe = np.where(valid_col_count > 0, col_min, 0.0)

    row_delta = np.where(
        valid,
        np.maximum(safe - row_min_safe[:, None], 0.0),
        0.0,
    )
    col_delta = np.where(
        valid,
        np.maximum(safe - col_min_safe[None, :], 0.0),
        0.0,
    )

    row_second = np.full(n, np.inf, dtype=float)
    col_second = np.full(n, np.inf, dtype=float)

    rows_with_two = valid_count >= 2
    cols_with_two = valid_col_count >= 2

    if np.any(rows_with_two):
        masked_rows = np.where(valid[rows_with_two], d[rows_with_two], np.inf)
        row_second[rows_with_two] = np.partition(
            masked_rows,
            kth=1,
            axis=1,
        )[:, 1]

    if np.any(cols_with_two):
        masked_cols = np.where(valid[:, cols_with_two], d[:, cols_with_two], np.inf)
        col_second[cols_with_two] = np.partition(
            masked_cols,
            kth=1,
            axis=0,
        )[1, :]

    row_gap = np.where(
        rows_with_two,
        np.maximum(row_second - row_min_safe, 0.0),
        distance_scale,
    )
    col_gap = np.where(
        cols_with_two,
        np.maximum(col_second - col_min_safe, 0.0),
        distance_scale,
    )

    positive_row_gaps = row_gap[(row_gap > 0.0) & np.isfinite(row_gap)]
    positive_col_gaps = col_gap[(col_gap > 0.0) & np.isfinite(col_gap)]

    row_gap_scale = (
        float(np.median(positive_row_gaps))
        if positive_row_gaps.size
        else distance_scale
    )
    col_gap_scale = (
        float(np.median(positive_col_gaps))
        if positive_col_gaps.size
        else distance_scale
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

    temperature = max(distance_scale * 0.12, eps)
    logits = np.clip(row_delta / temperature, 0.0, 60.0)
    affinity = np.exp(-logits)
    affinity[~valid] = 0.0

    balanced = affinity
    for _ in range(96):
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

        if np.max(np.abs(balanced - previous)) < 1e-12:
            break

    score = balanced * np.power(regret + eps, 1.3)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
