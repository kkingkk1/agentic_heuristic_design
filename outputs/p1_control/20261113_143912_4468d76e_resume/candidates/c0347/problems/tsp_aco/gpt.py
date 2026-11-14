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

    symmetric_d = np.zeros_like(d)
    symmetric_d[valid] = 0.5 * (d[valid] + d.T[valid])

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
    degree_fraction = degree_safe / max(n - 1, 1)

    # Vectorized nearest-neighbor ranks.
    order = np.argsort(effective_d, axis=1, kind="stable")
    sorted_d = np.take_along_axis(effective_d, order, axis=1)

    neighbor_count = np.maximum(degree.astype(int), 1)
    k = np.minimum(
        np.maximum(2, np.ceil(np.sqrt(neighbor_count)).astype(int)),
        neighbor_count,
    )

    local_scale = np.empty(n, dtype=float)
    for i in range(n):
        local_scale[i] = np.median(sorted_d[i, :k[i]])

    local_scale = np.where(
        np.isfinite(local_scale),
        np.maximum(local_scale, eps),
        global_scale,
    )

    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    ranks[~valid] = np.inf

    normalized_rank = ranks / degree_safe[:, None]
    rank_score = np.maximum(1.0 - normalized_rank, eps)
    mutual_rank = np.sqrt(rank_score * rank_score.T)
    mutual_rank = np.where(valid, mutual_rank, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_length = effective_d / np.maximum(pair_scale, eps)
    relative_length = np.minimum(relative_length, 1e6)

    endpoint_relative = np.maximum(
        effective_d / local_scale[:, None],
        effective_d / local_scale[None, :],
    )
    endpoint_relative = np.minimum(endpoint_relative, 40.0)

    global_relative = np.minimum(effective_d / global_scale, 40.0)

    sparsity = 1.0 / np.sqrt(
        np.maximum(degree_fraction[:, None], eps)
        * np.maximum(degree_fraction[None, :], eps)
    )
    sparsity = np.minimum(sparsity, 8.0)

    temperature = 0.80 + 0.60 * np.minimum(sparsity - 1.0, 3.0)
    long_edge_penalty = np.minimum(relative_length / temperature, 18.0)

    candidate_limit = 1.9 * np.maximum(
        local_scale[:, None],
        local_scale[None, :],
    )
    reciprocal_candidate = (
        valid
        & (symmetric_d <= candidate_limit)
        & (symmetric_d <= candidate_limit.T)
    )

    # Reward edges that are globally short and locally plausible.
    log_score = (
        1.20 * np.log(np.maximum(global_scale / effective_d, eps))
        + 1.55 * np.log(mutual_rank)
        - 1.15 * long_edge_penalty
        - 0.22 * global_relative
        - 0.48 * endpoint_relative
        + 0.55 * reciprocal_candidate.astype(float)
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
