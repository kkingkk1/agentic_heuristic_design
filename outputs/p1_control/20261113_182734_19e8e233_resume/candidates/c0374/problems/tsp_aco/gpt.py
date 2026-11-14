import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diverse, bidirectional, locally informed priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]

    if positive.size:
        scale = float(np.median(positive))
    else:
        finite = d[valid]
        scale = float(np.median(finite)) if finite.size else 1.0

    scale = max(scale, eps)
    masked = np.where(valid, d, np.inf)

    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)

    safe_row_best = np.where(np.isfinite(row_best), row_best, 0.0)
    safe_col_best = np.where(np.isfinite(col_best), col_best, 0.0)

    row_excess = np.where(
        valid,
        np.maximum(d - safe_row_best[:, None], 0.0),
        0.0,
    )
    col_excess = np.where(
        valid,
        np.maximum(d - safe_col_best[None, :], 0.0),
        0.0,
    )

    row_gaps = np.partition(masked, 1, axis=1)[:, 1] - row_best
    col_gaps = np.partition(masked, 1, axis=0)[1, :] - col_best

    row_gaps = row_gaps[np.isfinite(row_gaps) & (row_gaps > 0.0)]
    col_gaps = col_gaps[np.isfinite(col_gaps) & (col_gaps > 0.0)]

    row_gap_scale = float(np.median(row_gaps)) if row_gaps.size else scale
    col_gap_scale = float(np.median(col_gaps)) if col_gaps.size else scale
    row_gap_scale = max(row_gap_scale, 0.5 * scale, eps)
    col_gap_scale = max(col_gap_scale, 0.5 * scale, eps)

    temperature = max(0.40 * scale, eps)

    with np.errstate(over="ignore", invalid="ignore"):
        geometric = np.exp(
            -0.5
            * (
                row_excess / temperature
                + col_excess / temperature
            )
        )
        regret = np.exp(
            -0.5
            * (
                row_excess / row_gap_scale
                + col_excess / col_gap_scale
            )
        )

    geometric[~valid] = 0.0
    regret[~valid] = 0.0

    row_rank = np.sum(
        valid & (d <= safe_row_best[:, None] + row_excess),
        axis=1,
    )
    col_rank = np.sum(
        valid & (d <= safe_col_best[None, :] + col_excess),
        axis=0,
    )

    row_rank = np.maximum(row_rank, 1)
    col_rank = np.maximum(col_rank, 1)

    rank_reward = (
        1.0 / np.sqrt(row_rank[:, None] * col_rank[None, :])
    )
    rank_reward[~valid] = 0.0

    row_nearest = np.zeros((n, n), dtype=bool)
    col_nearest = np.zeros((n, n), dtype=bool)

    row_nearest[np.arange(n), np.argmin(masked, axis=1)] = True
    col_nearest[np.argmin(masked, axis=0), np.arange(n)] = True

    row_nearest &= valid
    col_nearest &= valid

    mutual_bonus = np.ones((n, n), dtype=float)
    mutual_bonus[row_nearest & col_nearest] = 1.28
    mutual_bonus[row_nearest ^ col_nearest] = 1.08
    mutual_bonus[~valid] = 0.0

    raw = (
        geometric
        * np.power(np.maximum(regret, eps), 0.38)
        * (0.78 + 0.22 * rank_reward)
        * mutual_bonus
    )
    raw[~valid] = 0.0
    raw[~np.isfinite(raw)] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(24):
            row_mass = raw @ col_scale
            new_row_scale = np.zeros_like(row_scale)
            np.divide(
                1.0,
                row_mass,
                out=new_row_scale,
                where=row_mass > 0.0,
            )

            col_mass = raw.T @ new_row_scale
            new_col_scale = np.zeros_like(col_scale)
            np.divide(
                1.0,
                col_mass,
                out=new_col_scale,
                where=col_mass > 0.0,
            )

            row_scale = new_row_scale
            col_scale = new_col_scale

    balanced = raw * row_scale[:, None] * col_scale[None, :]
    balanced[~valid] = 0.0
    balanced[~np.isfinite(balanced)] = 0.0

    maximum = float(balanced.max())
    if maximum > 0.0:
        balanced /= maximum

    raw_maximum = float(raw.max())
    if raw_maximum > 0.0:
        raw_normalized = raw / raw_maximum
    else:
        raw_normalized = np.zeros_like(raw)

    score = 0.82 * balanced + 0.14 * raw_normalized

    valid_floor = 0.04
    score += valid_floor * valid.astype(float)

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
