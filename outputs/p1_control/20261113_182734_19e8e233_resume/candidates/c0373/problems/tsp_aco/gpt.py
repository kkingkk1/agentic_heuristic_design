import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return lightweight, regret-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not valid.any():
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)
    masked = np.where(valid, d, np.inf)

    row_best = masked.min(axis=1)
    col_best = masked.min(axis=0)

    row_best_safe = np.where(np.isfinite(row_best), row_best, 0.0)
    col_best_safe = np.where(np.isfinite(col_best), col_best, 0.0)

    row_excess = np.where(
        valid,
        np.maximum(d - row_best_safe[:, None], 0.0),
        0.0,
    )
    col_excess = np.where(
        valid,
        np.maximum(d - col_best_safe[None, :], 0.0),
        0.0,
    )

    row_positive = row_excess[(row_excess > 0.0) & valid]
    col_positive = col_excess[(col_excess > 0.0) & valid]

    row_gap_scale = (
        float(np.median(row_positive))
        if row_positive.size
        else global_scale
    )
    col_gap_scale = (
        float(np.median(col_positive))
        if col_positive.size
        else global_scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    with np.errstate(over="ignore", invalid="ignore"):
        regret = np.exp(
            -0.5
            * (
                row_excess / row_gap_scale
                + col_excess / col_gap_scale
            )
        )

    temperature = max(0.30 * global_scale, eps)
    with np.errstate(over="ignore", invalid="ignore"):
        distance_affinity = np.exp(
            -np.minimum(
                (row_excess + col_excess) / (2.0 * temperature),
                60.0,
            )
        )

    row_count = np.maximum(valid.sum(axis=1), 1)
    col_count = np.maximum(valid.sum(axis=0), 1)

    row_rank_like = np.exp(
        -0.45 * np.minimum(row_excess / row_gap_scale, 12.0)
    )
    col_rank_like = np.exp(
        -0.45 * np.minimum(col_excess / col_gap_scale, 12.0)
    )

    score = (
        distance_affinity
        * regret
        * np.sqrt(row_rank_like * col_rank_like)
    )
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    row_mass = score.sum(axis=1)
    col_mass = score.sum(axis=0)

    row_mass = np.maximum(row_mass, eps)
    col_mass = np.maximum(col_mass, eps)

    degree_correction = 1.0 / np.sqrt(
        (row_mass / row_count)[:, None]
        * (col_mass / col_count)[None, :]
    )

    score *= degree_correction
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
