import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return low-cost, locally balanced priors for directed TSP edges."""
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

    tiny = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, tiny)

    safe = np.where(valid, d, np.inf)

    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    row_rank = np.argsort(np.argsort(np.where(valid, d, np.inf), axis=1), axis=1)
    col_rank = np.argsort(np.argsort(np.where(valid, d, np.inf), axis=0), axis=0)

    row_gap = np.partition(safe, kth=1, axis=1)[:, 1] - row_min
    col_gap = np.partition(safe, kth=1, axis=0)[1, :] - col_min

    row_gap = np.maximum(row_gap, 0.0)
    col_gap = np.maximum(col_gap, 0.0)

    positive_row_gap = row_gap[row_gap > 0.0]
    positive_col_gap = col_gap[col_gap > 0.0]

    row_gap_scale = (
        float(np.median(positive_row_gap))
        if positive_row_gap.size
        else scale
    )
    col_gap_scale = (
        float(np.median(positive_col_gap))
        if positive_col_gap.size
        else scale
    )

    row_gap_scale = max(row_gap_scale, tiny)
    col_gap_scale = max(col_gap_scale, tiny)

    row_delta = np.maximum(d - row_min[:, None], 0.0)
    col_delta = np.maximum(d - col_min[None, :], 0.0)

    temperature = max(0.30 * scale, tiny)
    local_affinity = np.exp(
        -np.clip(0.5 * row_delta / temperature, 0.0, 50.0)
        -np.clip(0.5 * col_delta / temperature, 0.0, 50.0)
    )

    row_regret = np.exp(
        -np.clip(row_delta / row_gap_scale, 0.0, 50.0)
    )
    col_regret = np.exp(
        -np.clip(col_delta / col_gap_scale, 0.0, 50.0)
    )
    regret = np.sqrt(row_regret * col_regret)

    rank_signal = np.exp(
        -0.5 * (
            row_rank.astype(float) / max(n - 1, 1)
            + col_rank.astype(float) / max(n - 1, 1)
        )
    )

    score = local_affinity * np.power(regret + tiny, 0.8)
    score *= 0.75 + 0.25 * rank_signal
    score[~valid] = 0.0

    for _ in range(8):
        row_sum = score.sum(axis=1, keepdims=True)
        score = np.divide(
            score,
            row_sum,
            out=np.zeros_like(score),
            where=row_sum > 0.0,
        )

        col_sum = score.sum(axis=0, keepdims=True)
        score = np.divide(
            score,
            col_sum,
            out=np.zeros_like(score),
            where=col_sum > 0.0,
        )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
