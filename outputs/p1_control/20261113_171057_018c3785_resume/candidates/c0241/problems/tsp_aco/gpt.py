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
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe = np.where(valid, d, np.inf)
    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    row_delta = np.zeros_like(d)
    col_delta = np.zeros_like(d)

    row_delta[valid] = np.maximum(
        d[valid] - np.repeat(row_min, n)[valid.ravel()],
        0.0,
    )
    col_delta[valid] = np.maximum(
        d[valid] - np.tile(col_min, n)[valid.ravel()],
        0.0,
    )

    row_gap = np.zeros(n, dtype=float)
    col_gap = np.zeros(n, dtype=float)

    if n > 2:
        row_second = np.partition(safe, 1, axis=1)[:, 1]
        col_second = np.partition(safe, 1, axis=0)[1, :]

        row_gap = np.where(
            np.isfinite(row_min) & np.isfinite(row_second),
            np.maximum(row_second - row_min, 0.0),
            0.0,
        )
        col_gap = np.where(
            np.isfinite(col_min) & np.isfinite(col_second),
            np.maximum(col_second - col_min, 0.0),
            0.0,
        )

    row_gap_scale = max(
        float(np.median(row_gap[row_gap > 0.0]))
        if np.any(row_gap > 0.0)
        else scale,
        np.finfo(float).eps,
    )
    col_gap_scale = max(
        float(np.median(col_gap[col_gap > 0.0]))
        if np.any(col_gap > 0.0)
        else scale,
        np.finfo(float).eps,
    )

    temperature = max(0.42 * scale, np.finfo(float).eps)

    row_affinity = np.exp(-np.clip(row_delta / temperature, 0.0, 60.0))
    col_affinity = np.exp(-np.clip(col_delta / temperature, 0.0, 60.0))

    regret_temperature = 1.15
    row_regret = np.exp(
        -np.clip(
            row_delta / (regret_temperature * row_gap_scale),
            0.0,
            60.0,
        )
    )
    col_regret = np.exp(
        -np.clip(
            col_delta / (regret_temperature * col_gap_scale),
            0.0,
            60.0,
        )
    )

    row_unique = row_gap / (row_gap + scale)
    col_unique = col_gap / (col_gap + scale)

    top_k = min(2, n - 1)
    row_threshold = np.partition(safe, top_k - 1, axis=1)[:, top_k - 1]
    col_threshold = np.partition(safe, top_k - 1, axis=0)[top_k - 1, :]

    mutual = (
        valid
        & (d <= row_threshold[:, None])
        & (d <= col_threshold[None, :])
    )

    score = np.sqrt(row_affinity * col_affinity)
    score *= np.sqrt(row_regret * col_regret)
    score *= 1.0 + 0.30 * mutual.astype(float)
    score *= 1.0 + 0.18 * np.sqrt(
        row_unique[:, None] * col_unique[None, :]
    )

    score[~valid] = 0.0

    destination_pressure = score.sum(axis=0, keepdims=True)
    score = np.divide(
        score,
        np.sqrt(np.maximum(destination_pressure, np.finfo(float).eps)),
        out=np.zeros_like(score),
        where=destination_pressure > 0.0,
    )

    source_total = score.sum(axis=1, keepdims=True)
    score = np.divide(
        score,
        source_total,
        out=np.zeros_like(score),
        where=source_total > 0.0,
    )

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
