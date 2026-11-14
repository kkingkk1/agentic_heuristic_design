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

    positive_values = symmetric_d[valid & (symmetric_d > 0.0)]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    effective_d = np.where(
        valid,
        np.maximum(symmetric_d, eps * global_scale),
        np.inf,
    )

    degree = valid.sum(axis=1).astype(float)
    degree_fraction = degree / max(n - 1, 1)

    local_scale = np.full(n, global_scale, dtype=float)
    nearest_rank = np.full((n, n), np.inf, dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        distances = effective_d[i, neighbors]
        order = np.argsort(distances, kind="stable")
        neighbors = neighbors[order]
        distances = distances[order]

        k = min(
            max(2, int(np.ceil(np.sqrt(neighbors.size)))),
            neighbors.size,
        )
        local_scale[i] = max(float(np.median(distances[:k])), eps)
        nearest_rank[i, neighbors] = np.arange(neighbors.size, dtype=float)

    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_length = effective_d / np.maximum(pair_scale, eps)
    relative_length = np.minimum(relative_length, 1e6)

    global_relative = effective_d / global_scale
    global_relative = np.minimum(global_relative, 1e6)

    valid_degree = np.maximum(degree, 1.0)
    normalized_rank = nearest_rank / valid_degree[:, None]
    rank_score = np.maximum(1.0 - normalized_rank, eps)

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

    temperature = 0.80 + 0.62 * np.minimum(sparsity - 1.0, 3.5)
    adaptive_penalty = np.minimum(relative_length / temperature, 18.0)

    candidate_limit = 1.9 * np.maximum(
        local_scale[:, None],
        local_scale[None, :],
    )
    reciprocal_candidate = (
        valid
        & (symmetric_d <= candidate_limit)
        & (symmetric_d <= candidate_limit.T)
    )

    log_score = (
        0.98 * np.log(np.maximum(global_scale / effective_d, eps))
        + 1.48 * np.log(mutual_rank)
        - 0.96 * adaptive_penalty
        - 0.24 * np.minimum(global_relative, 24.0)
        - 0.54 * np.minimum(endpoint_relative, 20.0)
        + 0.50 * reciprocal_candidate.astype(float)
        + 0.15 * np.log(sparsity)
    )

    log_score += np.where(valid, 0.0, -np.inf)
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
