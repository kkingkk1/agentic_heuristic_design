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

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe = np.where(valid, d, np.inf)

    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    rank = min(1, n - 1)
    row_second = np.partition(safe, kth=rank, axis=1)[:, rank]
    col_second = np.partition(safe, kth=rank, axis=0)[rank, :]

    row_gap = np.maximum(row_second - row_min, 0.0)
    col_gap = np.maximum(col_second - col_min, 0.0)

    row_gap_scale = (
        float(np.median(row_gap[row_gap > 0.0]))
        if np.any(row_gap > 0.0)
        else scale
    )
    col_gap_scale = (
        float(np.median(col_gap[col_gap > 0.0]))
        if np.any(col_gap > 0.0)
        else scale
    )

    tiny = np.finfo(float).eps
    row_gap_scale = max(row_gap_scale, tiny)
    col_gap_scale = max(col_gap_scale, tiny)

    row_regret = np.exp(
        -np.maximum(d - row_min[:, None], 0.0) / row_gap_scale
    )
    col_regret = np.exp(
        -np.maximum(d - col_min[None, :], 0.0) / col_gap_scale
    )
    regret = np.sqrt(row_regret * col_regret)

    centered = np.where(valid, d - row_min[:, None], 0.0)
    temperature = max(scale * 0.18, tiny)

    affinity = np.exp(-np.clip(centered / temperature, 0.0, 60.0))
    affinity[~valid] = 0.0

    balanced = affinity.copy()
    for _ in range(96):
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

    score = balanced * np.power(regret + tiny, 1.2)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
