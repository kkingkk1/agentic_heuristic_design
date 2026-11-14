import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    d = 0.5 * (d + d.T)
    valid = np.isfinite(d) & (d > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid]
    global_scale = max(float(np.median(positive)), eps)
    replacement = max(float(np.max(positive)) * 10.0, global_scale)

    work = np.where(valid, d, replacement)
    np.fill_diagonal(work, replacement)

    neighbor_count = min(2, n - 1)
    sorted_work = np.sort(work, axis=1)
    local_scale = np.mean(
        sorted_work[:, :neighbor_count],
        axis=1,
    )
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_length = work / geometric_scale
    global_length = work / global_scale

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    endpoint_rank = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(endpoint_rank * endpoint_rank.T)

    scale_gap = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_compatibility = np.exp(-0.4 * np.minimum(scale_gap, 50.0))

    local_affinity = np.exp(
        -np.minimum(work / local_scale[:, None], 50.0)
    )
    neighborhood_mass = np.sum(local_affinity * valid, axis=1)

    crowding_penalty = 1.0 / np.sqrt(
        (1.0 + neighborhood_mass[:, None])
        * (1.0 + neighborhood_mass[None, :])
    )

    log_score = (
        -0.95 * normalized_length
        -0.25 * global_length
        +1.25 * np.log(np.maximum(mutual_rank, eps))
        +0.45 * np.log(np.maximum(scale_compatibility, eps))
        +0.25 * np.log(np.maximum(crowding_penalty, eps))
    )

    finite_scores = valid & np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    log_score = np.where(finite_scores, log_score, -np.inf)
    log_score -= np.max(log_score[finite_scores])

    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores], -745.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
