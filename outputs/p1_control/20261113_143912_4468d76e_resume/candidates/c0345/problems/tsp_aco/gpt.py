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

    eps = np.finfo(float).eps

    symmetric_d = np.zeros_like(d, dtype=float)
    symmetric_d[valid] = 0.5 * (d[valid] + d.T[valid])
    symmetric_d = np.maximum(symmetric_d, 0.0)

    positive = symmetric_d[valid & (symmetric_d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    effective_d = np.where(
        valid,
        np.maximum(symmetric_d, eps * global_scale),
        np.inf,
    )

    degree = valid.sum(axis=1).astype(float)
    degree_safe = np.maximum(degree, 1.0)
    degree_fraction = degree / max(n - 1, 1)

    local_scale = np.full(n, global_scale, dtype=float)
    rank = np.full((n, n), np.inf, dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        distances = effective_d[i, neighbors]
        order = np.argsort(distances, kind="stable")
        neighbors = neighbors[order]
        distances = distances[order]

        k = min(neighbors.size, max(2, int(np.ceil(np.sqrt(neighbors.size)))))
        local_scale[i] = max(float(np.median(distances[:k])), eps)
        rank[i, neighbors] = np.arange(neighbors.size, dtype=float)

    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    local_relative = np.minimum(effective_d / pair_scale, 1e6)
    global_relative = np.minimum(effective_d / global_scale, 1e6)

    rank_fraction = rank / degree_safe[:, None]
    rank_score = np.maximum(1.0 - rank_fraction, eps)
    mutual_rank = np.sqrt(rank_score * rank_score.T)
    mutual_rank = np.where(valid, mutual_rank, eps)

    endpoint_relative = np.maximum(
        effective_d / local_scale[:, None],
        effective_d / local_scale[None, :],
    )
    endpoint_relative = np.minimum(endpoint_relative, 40.0)

    sparsity = 1.0 / np.sqrt(
        np.maximum(degree_fraction[:, None], eps)
        * np.maximum(degree_fraction[None, :], eps)
    )
    sparsity = np.minimum(sparsity, 8.0)

    temperature = 0.80 + 0.50 * np.minimum(sparsity - 1.0, 3.0)
    soft_penalty = np.minimum(local_relative / temperature, 16.0)

    candidate_limit = 1.75 * np.maximum(
        local_scale[:, None],
        local_scale[None, :],
    )
    candidate = valid & (symmetric_d <= candidate_limit)

    endpoint_rank = np.minimum(rank, rank.T)
    bridge = np.where(
        valid & (endpoint_rank >= np.minimum(degree[:, None], degree[None, :]) - 1.0),
        1.0,
        0.0,
    )

    log_score = (
        1.05 * np.log(np.maximum(global_scale / effective_d, eps))
        + 1.45 * np.log(mutual_rank)
        - 0.92 * soft_penalty
        - 0.32 * np.minimum(global_relative, 24.0)
        - 0.48 * np.minimum(endpoint_relative, 20.0)
        + 0.48 * candidate.astype(float)
        + 0.22 * bridge
        + 0.16 * np.log(sparsity)
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
