import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    valid = np.isfinite(d) & (d > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = d[valid]
    global_scale = max(float(np.median(positive)), eps)
    replacement = max(float(np.max(positive)) * 10.0, global_scale)

    work = np.where(valid, d, replacement)
    np.fill_diagonal(work, replacement)

    neighbor_count = min(2, n - 1)
    nearest = np.partition(
        work,
        neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]
    local_scale = np.maximum(np.mean(nearest, axis=1), eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = work / endpoint_scale
    global_length = work / global_scale

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    reciprocal_rank = 1.0 / (1.0 + ranks + ranks.T)

    scale_gap = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_gap = np.minimum(scale_gap, 20.0)

    local_density = np.mean(
        work <= 1.5 * local_scale[:, None],
        axis=1,
    )
    density_gap = np.abs(
        local_density[:, None] - local_density[None, :]
    )

    log_score = (
        1.00 * np.log(global_scale / np.maximum(work, eps))
        + 1.65 * np.log(reciprocal_rank)
        - 0.85 * normalized_length
        - 0.20 * global_length
        - 0.18 * scale_gap
        - 0.12 * density_gap * np.minimum(normalized_length, 3.0)
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    log_score -= np.max(log_score[finite_scores])
    score = np.exp(np.clip(log_score, -745.0, 0.0))
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
