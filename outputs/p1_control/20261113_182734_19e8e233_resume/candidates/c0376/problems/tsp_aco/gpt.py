import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return inexpensive two-sided regret-aware priors for directed TSP edges."""
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
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    masked = np.where(valid, d, np.inf)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_min = np.min(masked, axis=1)
    col_min = np.min(masked, axis=0)

    row_min_safe = np.where(row_count > 0, row_min, 0.0)
    col_min_safe = np.where(col_count > 0, col_min, 0.0)

    row_delta = np.where(
        valid,
        np.maximum(d - row_min_safe[:, None], 0.0),
        0.0,
    )
    col_delta = np.where(
        valid,
        np.maximum(d - col_min_safe[None, :], 0.0),
        0.0,
    )

    row_second = np.full(n, np.inf, dtype=float)
    col_second = np.full(n, np.inf, dtype=float)

    rows_with_two = row_count >= 2
    cols_with_two = col_count >= 2

    if np.any(rows_with_two):
        row_values = masked[rows_with_two]
        row_second[rows_with_two] = np.partition(
            row_values,
            kth=1,
            axis=1,
        )[:, 1]

    if np.any(cols_with_two):
        col_values = masked[:, cols_with_two]
        col_second[cols_with_two] = np.partition(
            col_values,
            kth=1,
            axis=0,
        )[1, :]

    row_gap = np.where(
        rows_with_two,
        np.maximum(row_second - row_min_safe, 0.0),
        global_scale,
    )
    col_gap = np.where(
        cols_with_two,
        np.maximum(col_second - col_min_safe, 0.0),
        global_scale,
    )

    row_gap_scale_values = row_gap[
        (row_gap > 0.0) & np.isfinite(row_gap)
    ]
    col_gap_scale_values = col_gap[
        (col_gap > 0.0) & np.isfinite(col_gap)
    ]

    row_gap_scale = (
        float(np.median(row_gap_scale_values))
        if row_gap_scale_values.size
        else global_scale
    )
    col_gap_scale = (
        float(np.median(col_gap_scale_values))
        if col_gap_scale_values.size
        else global_scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_distance_signal = np.exp(
        -np.clip(row_delta / (global_scale * 0.18), 0.0, 50.0)
    )
    col_distance_signal = np.exp(
        -np.clip(col_delta / (global_scale * 0.18), 0.0, 50.0)
    )

    row_regret_signal = np.exp(
        -np.clip(row_delta / row_gap_scale, 0.0, 50.0)
    )
    col_regret_signal = np.exp(
        -np.clip(col_delta / col_gap_scale, 0.0, 50.0)
    )

    distance_affinity = np.sqrt(
        row_distance_signal * col_distance_signal
    )
    regret_affinity = np.sqrt(
        row_regret_signal * col_regret_signal
    )

    row_best = valid & (
        row_delta <= 0.5 * np.maximum(row_gap[:, None], global_scale * eps)
    )
    col_best = valid & (
        col_delta <= 0.5 * np.maximum(col_gap[None, :], global_scale * eps)
    )

    local_competition_bonus = (
        1.0
        + 0.10 * row_best.astype(float)
        + 0.10 * col_best.astype(float)
        + 0.15 * (row_best & col_best).astype(float)
    )

    score = (
        distance_affinity
        * (0.65 + 0.35 * regret_affinity)
        * local_competition_bonus
    )
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    row_mass = score.sum(axis=1, keepdims=True)
    col_mass = score.sum(axis=0, keepdims=True)
    mass_correction = np.sqrt(
        np.maximum(row_mass, eps) * np.maximum(col_mass, eps)
    )
    diversity_signal = np.divide(
        score,
        mass_correction,
        out=np.zeros_like(score),
        where=mass_correction > 0.0,
    )

    score = 0.85 * score + 0.15 * diversity_signal
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
