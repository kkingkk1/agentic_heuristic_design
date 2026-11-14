import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    finite_positive = np.isfinite(d) & (d > 0)

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    positive_values = d[finite_positive]
    eps = np.finfo(float).eps
    global_scale = max(float(np.median(positive_values)), eps)
    replacement = max(float(np.max(positive_values)) * 10.0, eps)

    work = np.where(finite_positive, d, replacement)
    np.fill_diagonal(work, replacement)

    nearest = np.partition(work, 1, axis=1)[:, :2]
    local_scale = np.maximum(nearest.mean(axis=1), eps)

    normalized_length = work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)

    ranks += ranks.T
    np.log1p(ranks, out=ranks)

    log_score = -0.72 * normalized_length
    log_score -= 0.22 * (work / global_scale)
    log_score -= 1.9 * ranks

    log_density = np.log(global_scale / local_scale)
    density_gap = np.abs(
        log_density[:, None] - log_density[None, :]
    )
    density_gap /= (
        1.0
        + np.abs(
            log_density[:, None] + log_density[None, :]
        )
    )

    log_score -= 0.55 * density_gap
    np.clip(log_score, -745.0, 0.0, out=log_score)
    np.exp(log_score, out=log_score)

    floor = 0.35 / (1.0 + 0.5 * density_gap)
    modulation = np.exp(-0.35 * normalized_length)
    modulation *= 1.0 - floor
    modulation += floor

    score = log_score * modulation
    score[~finite_positive] = 0.0
    np.fill_diagonal(score, 0.0)

    score += score.T
    score *= 0.5

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
