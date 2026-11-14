import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diversified, globally balanced priors for directed TSP edges."""
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

    row_rank_order = np.argsort(safe, axis=1)
    col_rank_order = np.argsort(safe, axis=0)

    nearest_col = row_rank_order[:, 0]
    nearest_row = col_rank_order[0, :]

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest_col[i]
        if valid[i, j] and nearest_row[j] == i:
            mutual_nearest[i, j] = 1.0

    k = min(4, n - 1)
    rank_bonus = np.zeros((n, n), dtype=float)

    for rank in range(k):
        weight = 1.0 / (rank + 1.0)
        row_indices = np.arange(n)
        cols = row_rank_order[:, rank]
        admissible = valid[row_indices, cols]
        rank_bonus[row_indices[admissible], cols[admissible]] += weight

    rank_max = float(np.max(rank_bonus))
    if rank_max > 0.0:
        rank_bonus /= rank_max

    row_second = (
        np.partition(safe, kth=1, axis=1)[:, 1]
        if n > 2
        else np.max(safe, axis=1)
    )
    col_second = (
        np.partition(safe, kth=1, axis=0)[1, :]
        if n > 2
        else np.max(safe, axis=0)
    )

    row_gap = np.maximum(row_second - row_min, 0.0)
    col_gap = np.maximum(col_second - col_min, 0.0)

    row_gap_positive = row_gap[row_gap > 0.0]
    col_gap_positive = col_gap[col_gap > 0.0]

    row_gap_scale = (
        float(np.median(row_gap_positive))
        if row_gap_positive.size
        else scale
    )
    col_gap_scale = (
        float(np.median(col_gap_positive))
        if col_gap_positive.size
        else scale
    )

    row_gap_scale = max(row_gap_scale, scale * 0.05, tiny)
    col_gap_scale = max(col_gap_scale, scale * 0.05, tiny)

    row_regret = np.exp(
        -np.maximum(d - row_min[:, None], 0.0) / row_gap_scale
    )
    col_regret = np.exp(
        -np.maximum(d - col_min[None, :], 0.0) / col_gap_scale
    )
    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    centered = np.maximum(d - row_min[:, None], 0.0)
    relative_distance = np.exp(-np.clip(centered / (scale * 0.8), 0.0, 50.0))
    relative_distance[~valid] = 0.0

    temperature = max(scale * 0.30, tiny)
    affinity = np.exp(-np.clip(centered / temperature, 0.0, 60.0))
    affinity[~valid] = 0.0

    valid_count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    row_mean = np.divide(
        np.sum(np.where(valid, affinity, 0.0), axis=1, keepdims=True),
        valid_count,
    )
    adaptive_floor = 0.035 * np.maximum(row_mean, tiny)

    seeded = affinity + adaptive_floor * valid.astype(float)

    balanced = seeded.copy()
    for _ in range(48):
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

    balanced = 0.72 * balanced + 0.28 * seeded

    local_signal = (
        0.50 * rank_bonus
        + 0.25 * mutual_nearest
        + 0.25 * relative_distance
    )

    score = (
        0.62 * balanced
        + 0.20 * regret
        + 0.13 * local_signal
        + 0.05 * affinity
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
