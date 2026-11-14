import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    valid = np.isfinite(d) & (d > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    replacement = max(float(np.max(positive_values)) * 10.0, eps)

    work = np.where(valid, d, replacement)
    np.fill_diagonal(work, replacement)

    k = min(2, n - 1)
    nearest = np.partition(work, k - 1, axis=1)[:, :k]

    if k == 1:
        local_scale = nearest[:, 0]
    else:
        local_scale = 0.8 * nearest[:, 0] + 0.2 * nearest[:, 1]

    local_scale = np.maximum(local_scale, eps)
    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_length = work / geometric_scale
    absolute_length = work / global_scale

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_support = 1.0 / (1.0 + ranks + ranks.T)

    log_density = np.log(global_scale / local_scale)
    density_gap = np.abs(
        log_density[:, None] - log_density[None, :]
    )
    density_gap /= (
        1.0
        + np.abs(log_density[:, None] + log_density[None, :])
    )

    log_score = (
        -1.15 * normalized_length
        -0.38 * absolute_length
        +0.55 * np.log(rank_support)
        -0.30 * density_gap
    )

    score = np.exp(np.clip(log_score, -745.0, 0.0))
    score *= np.exp(
        np.clip(-0.55 * normalized_length, -745.0, 0.0)
    )
    score *= np.exp(
        np.clip(-0.80 * density_gap, -745.0, 0.0)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
