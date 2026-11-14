import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d) & (d >= 0.0)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    eps = np.finfo(float).eps

    positive = d[valid & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    work = np.where(valid, np.maximum(d, 0.0), np.inf)

    positive_work = np.where(valid & (d > 0.0), d, np.inf)
    sorted_positive = np.sort(positive_work, axis=1)

    local_scale = np.full(n, global_scale, dtype=float)
    for i in range(n):
        row = sorted_positive[i]
        finite_row = row[np.isfinite(row)]
        if finite_row.size:
            local_scale[i] = finite_row[min(1, finite_row.size - 1)]

    local_scale = np.maximum(local_scale, eps)

    relative_length = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = np.where(valid, relative_length, np.inf)
    relative_length = np.minimum(relative_length, 1e6)

    rank_input = np.where(valid, work, np.inf)
    order = np.argsort(rank_input, axis=1, kind="stable")

    ranks = np.empty((n, n), dtype=float)
    row_index = np.arange(n)[:, None]
    ranks[row_index, order] = np.arange(n, dtype=float)[None, :]
    ranks = ranks / max(n - 1, 1)
    ranks = np.where(valid, ranks, 1.0)

    endpoint_rank_score = np.sqrt(
        np.maximum(1.0 - ranks, eps)
        * np.maximum(1.0 - ranks.T, eps)
    )

    reciprocal_term = global_scale / np.maximum(d, eps)
    reciprocal_term = np.minimum(reciprocal_term, 1e12)

    nearest_k = min(3, max(n - 1, 1))
    mutual_nearest = (
        (ranks <= nearest_k / max(n - 1, 1))
        & (ranks.T <= nearest_k / max(n - 1, 1))
        & valid
    )

    local_neighbor_mask = (
        valid
        & (
            work
            <= 1.35
            * np.minimum(local_scale[:, None], local_scale[None, :])
        )
    )
    local_density = local_neighbor_mask.mean(axis=1)

    congestion_penalty = (
        local_density[:, None] + local_density[None, :]
    ) * np.minimum(relative_length, 3.0)

    log_score = (
        1.00 * np.log(np.maximum(reciprocal_term, eps))
        + 0.88 * np.log(np.maximum(endpoint_rank_score, eps))
        - 0.92 * relative_length
        - 0.18 * d / global_scale
        - 0.08 * congestion_penalty
        + 0.20 * mutual_nearest.astype(float)
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.exp(np.clip(log_score - maximum, -745.0, 0.0))
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
