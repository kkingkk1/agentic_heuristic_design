import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diversity-preserving, rank-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)

    row_order = np.argsort(masked, axis=1, kind="stable")
    col_order = np.argsort(masked, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    base_rank = np.arange(n, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = base_rank
    col_rank[col_order, np.arange(n)[None, :]] = base_rank[:, None]

    row_rank[~valid] = np.inf
    col_rank[~valid] = np.inf

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    row_second = np.partition(masked, min(1, n - 1), axis=1)[:, min(1, n - 1)]
    col_second = np.partition(masked, min(1, n - 1), axis=0)[min(1, n - 1), :]

    row_gap = np.maximum(row_second - row_best, 0.0)
    col_gap = np.maximum(col_second - col_best, 0.0)

    finite_row_gap = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    finite_col_gap = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(finite_row_gap)) if finite_row_gap.size else scale
    )
    col_gap_scale = (
        float(np.median(finite_col_gap)) if finite_col_gap.size else scale
    )
    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_excess = np.maximum(d - row_best[:, None], 0.0)
    col_excess = np.maximum(d - col_best[None, :], 0.0)

    local_affinity = np.exp(-np.minimum(row_excess / (0.38 * scale), 55.0))
    global_affinity = np.exp(
        -np.minimum((row_excess + col_excess) / (0.95 * scale), 55.0)
    )

    local_affinity[~valid] = 0.0
    global_affinity[~valid] = 0.0

    row_rank_score = np.zeros_like(d)
    col_rank_score = np.zeros_like(d)

    row_rank_score[valid] = 1.0 / (1.0 + row_rank[valid])
    col_rank_score[valid] = 1.0 / (1.0 + col_rank[valid])

    row_rank_score /= max(float(np.max(row_rank_score)), eps)
    col_rank_score /= max(float(np.max(col_rank_score)), eps)

    mutual = np.zeros((n, n), dtype=float)
    nearest_col = np.argmin(masked, axis=1)
    nearest_row = np.argmin(masked, axis=0)

    for i in range(n):
        j = nearest_col[i]
        if np.isfinite(masked[i, j]) and nearest_row[j] == i:
            mutual[i, j] = 1.0

    third_neighbor = (row_rank == 2.0) | (col_rank == 2.0)
    rank_bonus = 1.0 + 0.20 * row_rank_score + 0.20 * col_rank_score
    rank_bonus += 0.18 * mutual
    rank_bonus += 0.04 * third_neighbor.astype(float)

    regret = np.exp(
        -0.5 * (
            row_excess / row_gap_scale
            + col_excess / col_gap_scale
        )
    )
    regret[~valid] = 0.0

    base = 0.68 * local_affinity + 0.32 * global_affinity
    score = base * rank_bonus * np.power(regret + eps, 0.72)
    score[~valid] = 0.0

    total = float(np.sum(score))
    if total <= 0.0 or not np.isfinite(total):
        return np.zeros((n, n), dtype=float)

    probability = score / total
    concentration = float(np.max(probability))
    floor_strength = np.clip(0.035 + 0.18 * concentration, 0.035, 0.12)

    valid_count = np.sum(valid, axis=1, keepdims=True).astype(float)
    floor = np.divide(
        valid.astype(float),
        np.maximum(valid_count, 1.0),
    )
    score = (1.0 - floor_strength) * score + floor_strength * floor

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(36):
            row_mass = score @ col_scale
            row_scale = np.divide(
                1.0,
                row_mass,
                out=np.zeros_like(row_mass),
                where=row_mass > 0.0,
            )

            col_mass = score.T @ row_scale
            col_scale = np.divide(
                1.0,
                col_mass,
                out=np.zeros_like(col_mass),
                where=col_mass > 0.0,
            )

    balanced = score * row_scale[:, None] * col_scale[None, :]

    distance_penalty = np.ones_like(d)
    finite_distances = d[valid]
    distance_scale = max(float(np.percentile(finite_distances, 80.0)), scale)
    distance_penalty[valid] = np.exp(
        -0.08 * np.maximum(d[valid] - distance_scale, 0.0) / distance_scale
    )

    result = balanced * distance_penalty
    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(np.max(result))
    if maximum > 0.0:
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result.astype(float, copy=False)
