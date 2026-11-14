import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    symmetric_d = np.zeros_like(d, dtype=float)
    symmetric_d[valid] = 0.5 * (d[valid] + d.T[valid])

    positive = symmetric_d[valid & (symmetric_d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    floor = eps * global_scale
    effective_d = np.full((n, n), np.inf, dtype=float)
    effective_d[valid] = np.maximum(symmetric_d[valid], floor)

    degree = valid.sum(axis=1).astype(float)
    degree_safe = np.maximum(degree, 1.0)
    degree_fraction = degree_safe / max(n - 1, 1)

    local_scale = np.full(n, global_scale, dtype=float)
    rank = np.full((n, n), np.inf, dtype=float)
    cutoff = np.ones(n, dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        distances = effective_d[i, neighbors]
        order = np.argsort(distances, kind="stable")
        ordered_neighbors = neighbors[order]
        ordered_distances = distances[order]

        k = min(
            ordered_neighbors.size,
            max(2, int(np.ceil(np.sqrt(ordered_neighbors.size)))),
        )

        positive_local = ordered_distances[ordered_distances > floor]
        if positive_local.size:
            local_scale[i] = max(
                float(np.median(positive_local[:k])),
                floor,
            )
        else:
            local_scale[i] = global_scale

        rank[i, ordered_neighbors] = np.arange(
            ordered_neighbors.size,
            dtype=float,
        )

        cutoff[i] = min(
            ordered_neighbors.size,
            max(3, int(np.ceil(np.sqrt(ordered_neighbors.size) * 1.5))),
        )

    local_scale = np.maximum(local_scale, floor)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, floor)

    global_relative = np.minimum(effective_d / global_scale, 1e8)
    local_relative = np.minimum(effective_d / pair_scale, 1e8)

    normalized_rank = rank / degree_safe[:, None]

    sparsity = 1.0 / np.sqrt(
        np.maximum(degree_fraction[:, None], eps)
        * np.maximum(degree_fraction[None, :], eps)
    )
    sparsity = np.minimum(sparsity, 8.0)

    # Sparse endpoints receive a wider rank temperature.
    endpoint_temperature = 0.8 + 0.45 * np.minimum(sparsity - 1.0, 4.0)
    rank_weight = np.exp(
        -np.minimum(normalized_rank / endpoint_temperature[:, None], 30.0)
    )
    rank_weight[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(rank_weight * rank_weight.T, eps)
    )
    mutual_rank[~valid] = eps

    reciprocal_candidate = (
        valid
        & (rank <= cutoff[:, None])
        & (rank.T <= cutoff[None, :])
    )

    locally_plausible = valid & (local_relative <= 2.5)
    candidate_bonus = (
        0.50 * reciprocal_candidate.astype(float)
        + 0.20 * locally_plausible.astype(float)
    )

    # Smooth penalties preserve useful long edges without letting them dominate.
    log_score = (
        -1.45 * np.log1p(global_relative)
        -1.20 * np.log1p(local_relative)
        +1.55 * np.log(mutual_rank)
        +candidate_bonus
        +0.16 * np.log(sparsity)
    )

    log_score[~valid] = -np.inf
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
