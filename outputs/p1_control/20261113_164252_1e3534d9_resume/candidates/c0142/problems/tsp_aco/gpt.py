import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diverse, locally ranked priors for directed TSP edges."""
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

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]
    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else float(np.median(finite_values))
        if finite_values.size
        else 1.0
    )
    scale = max(scale, np.finfo(float).eps)

    safe = np.where(valid, d, np.inf)

    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.empty_like(row_order)
    col_rank = np.empty_like(col_order)

    row_positions = np.arange(n)[None, :]
    col_positions = np.arange(n)[:, None]

    row_rank[row_positions, row_order] = np.arange(n)[None, :]
    col_rank[col_order, col_positions] = np.arange(n)[:, None]

    row_rank = np.where(valid, row_rank, n)
    col_rank = np.where(valid, col_rank, n)

    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    row_second = np.partition(safe, kth=min(1, n - 1), axis=1)[:, min(1, n - 1)]
    col_second = np.partition(safe, kth=min(1, n - 1), axis=0)[min(1, n - 1), :]

    row_gap = np.maximum(row_second - row_min, 0.0)
    col_gap = np.maximum(col_second - col_min, 0.0)

    positive_row_gaps = row_gap[row_gap > 0.0]
    positive_col_gaps = col_gap[col_gap > 0.0]

    row_gap_scale = (
        float(np.median(positive_row_gaps))
        if positive_row_gaps.size
        else scale
    )
    col_gap_scale = (
        float(np.median(positive_col_gaps))
        if positive_col_gaps.size
        else scale
    )

    row_gap_scale = max(row_gap_scale, np.finfo(float).eps)
    col_gap_scale = max(col_gap_scale, np.finfo(float).eps)

    row_selectivity = np.clip(row_gap / row_gap_scale, 0.0, 2.0)
    col_selectivity = np.clip(col_gap / col_gap_scale, 0.0, 2.0)

    centered_row = np.maximum(d - row_min[:, None], 0.0)
    centered_col = np.maximum(d - col_min[None, :], 0.0)

    temperature = max(0.35 * scale, np.finfo(float).eps)
    affinity = np.exp(
        -np.clip(0.5 * (centered_row + centered_col) / temperature, 0.0, 60.0)
    )
    affinity[~valid] = 0.0

    k = min(max(2, int(np.sqrt(n)) + 1), n - 1)
    local_mask = (row_rank < k) | (col_rank < k)

    rank_temperature = max(1.5, 0.5 * k)
    row_rank_score = np.exp(-row_rank / rank_temperature)
    col_rank_score = np.exp(-col_rank / rank_temperature)
    rank_score = np.sqrt(row_rank_score * col_rank_score)
    rank_score[~valid] = 0.0

    mutual_nearest = (row_rank == 0) & (col_rank == 0)
    near_mutual = (row_rank <= 1) & (col_rank <= 1)
    mutual_bonus = np.ones((n, n), dtype=float)
    mutual_bonus += 0.18 * near_mutual
    mutual_bonus += 0.12 * mutual_nearest

    row_regret = np.exp(
        -np.maximum(d - row_min[:, None], 0.0) / row_gap_scale
    )
    col_regret = np.exp(
        -np.maximum(d - col_min[None, :], 0.0) / col_gap_scale
    )
    regret = np.sqrt(row_regret * col_regret)

    selective_regret = (
        1.0
        + 0.16
        * regret
        * (
            row_selectivity[:, None] * (row_rank == 0)
            + col_selectivity[None, :] * (col_rank == 0)
        )
    )

    local_score = (
        0.50 * affinity
        + 0.34 * rank_score
        + 0.16 * np.sqrt(np.maximum(affinity * rank_score, 0.0))
    )

    tail_weight = np.where(local_mask, 1.0, 0.10)
    score = local_score * tail_weight * mutual_bonus * selective_regret
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
