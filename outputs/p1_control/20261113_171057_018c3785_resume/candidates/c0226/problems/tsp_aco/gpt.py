import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return fast, diversity-aware priors for directed TSP edges."""
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

    positive = d[valid & (d > 0.0)]
    eps = np.finfo(float).eps
    scale = max(
        float(np.median(positive)) if positive.size else 1.0,
        eps,
    )

    safe = np.where(valid, d, np.inf)
    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    row_delta = np.maximum(d - row_min[:, None], 0.0)
    col_delta = np.maximum(d - col_min[None, :], 0.0)
    row_delta[~valid] = 0.0
    col_delta[~valid] = 0.0

    if n > 2:
        row_second = np.partition(safe, 1, axis=1)[:, 1]
        col_second = np.partition(safe, 1, axis=0)[1, :]

        row_gap = np.maximum(row_second - row_min, 0.0)
        col_gap = np.maximum(col_second - col_min, 0.0)

        row_gap[~np.isfinite(row_gap)] = 0.0
        col_gap[~np.isfinite(col_gap)] = 0.0
    else:
        row_gap = np.zeros(n, dtype=float)
        col_gap = np.zeros(n, dtype=float)

    positive_row_gap = row_gap[row_gap > 0.0]
    positive_col_gap = col_gap[col_gap > 0.0]

    row_gap_scale = max(
        float(np.median(positive_row_gap))
        if positive_row_gap.size
        else scale,
        eps,
    )
    col_gap_scale = max(
        float(np.median(positive_col_gap))
        if positive_col_gap.size
        else scale,
        eps,
    )

    temperature = max(0.35 * scale, eps)

    decay = (
        (row_delta + col_delta) / (2.0 * temperature)
        + (row_delta / row_gap_scale + col_delta / col_gap_scale) / 2.0
    )
    score = np.exp(-np.clip(decay, 0.0, 60.0))

    row_unique = row_gap / (row_gap + scale)
    col_unique = col_gap / (col_gap + scale)
    score *= 1.0 + 0.20 * np.sqrt(
        row_unique[:, None] * col_unique[None, :]
    )

    top_k = min(2, n - 1)
    row_threshold = np.partition(safe, top_k - 1, axis=1)[:, top_k - 1]
    col_threshold = np.partition(safe, top_k - 1, axis=0)[top_k - 1, :]

    mutual = (
        valid
        & (d <= row_threshold[:, None])
        & (d <= col_threshold[None, :])
    )
    score *= 1.0 + 0.25 * mutual

    score[~valid] = 0.0

    destination_pressure = score.sum(axis=0, keepdims=True)
    score /= np.sqrt(np.maximum(destination_pressure, eps))

    source_total = score.sum(axis=1, keepdims=True)
    np.divide(
        score,
        source_total,
        out=score,
        where=source_total > 0.0,
    )

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
