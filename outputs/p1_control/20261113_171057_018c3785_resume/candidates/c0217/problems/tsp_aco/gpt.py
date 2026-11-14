import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, regret-aware stochastic priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    score = np.zeros((n, n), dtype=float)
    if not np.any(valid):
        return score

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe = np.where(valid, d, np.inf)

    row_best = safe.min(axis=1)
    col_best = safe.min(axis=0)

    row_second = np.partition(safe, 1, axis=1)[:, 1]
    col_second = np.partition(safe, 1, axis=0)[1, :]

    row_gap = np.maximum(row_second - row_best, 0.0)
    col_gap = np.maximum(col_second - col_best, 0.0)

    row_gap_values = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_gap_values = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(row_gap_values))
        if row_gap_values.size
        else scale
    )
    col_gap_scale = (
        float(np.median(col_gap_values))
        if col_gap_values.size
        else scale
    )

    eps = np.finfo(float).eps
    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_excess = np.where(valid, np.maximum(d - row_best[:, None], 0.0), 0.0)
    col_excess = np.where(valid, np.maximum(d - col_best[None, :], 0.0), 0.0)

    temperature = max(0.32 * scale, eps)

    row_affinity = np.exp(-np.minimum(row_excess / temperature, 60.0))
    col_affinity = np.exp(-np.minimum(col_excess / temperature, 60.0))

    row_regret = np.exp(-np.minimum(row_excess / row_gap_scale, 60.0))
    col_regret = np.exp(-np.minimum(col_excess / col_gap_scale, 60.0))

    affinity = np.sqrt(row_affinity * col_affinity)
    regret = np.sqrt(row_regret * col_regret)

    candidate_k = min(2, n - 1)
    row_threshold = np.partition(safe, candidate_k - 1, axis=1)[:, candidate_k - 1]
    col_threshold = np.partition(safe, candidate_k - 1, axis=0)[candidate_k - 1, :]

    mutual_candidate = (
        valid
        & (d <= row_threshold[:, None])
        & (d <= col_threshold[None, :])
    )

    row_uniqueness = row_gap / (row_gap + scale)
    col_uniqueness = col_gap / (col_gap + scale)
    uniqueness = np.sqrt(
        row_uniqueness[:, None] * col_uniqueness[None, :]
    )

    score = affinity * np.power(regret + eps, 0.8)
    score *= 1.0 + 0.22 * mutual_candidate.astype(float)
    score *= 1.0 + 0.18 * uniqueness
    score[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(24):
            row_mass = score @ col_scale
            row_scale.fill(0.0)
            np.divide(
                1.0,
                row_mass,
                out=row_scale,
                where=row_mass > 0.0,
            )

            col_mass = score.T @ row_scale
            col_scale.fill(0.0)
            np.divide(
                1.0,
                col_mass,
                out=col_scale,
                where=col_mass > 0.0,
            )

    score *= row_scale[:, None]
    score *= col_scale[None, :]
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
