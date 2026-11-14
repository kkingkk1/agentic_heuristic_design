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
    symmetric_d = np.maximum(symmetric_d, 0.0)

    positive = valid & (symmetric_d > 0.0)
    positive_values = symmetric_d[positive]

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

    # Second-neighbor local scale, with robust fallbacks.
    sorted_distances = np.sort(effective_d, axis=1)
    local_scale = sorted_distances[:, min(1, n - 1)]

    first_scale = sorted_distances[:, 0]
    row_mean = np.divide(
        np.sum(np.where(valid, effective_d, 0.0), axis=1),
        np.maximum(degree, 1.0),
    )

    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        np.where(np.isfinite(first_scale), first_scale, row_mean),
    )
    local_scale = np.where(
        local_scale > 0.0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = effective_d / np.maximum(pair_scale, eps)
    relative_length = np.minimum(relative_length, 1e6)

    global_relative = effective_d / global_scale
    global_relative = np.minimum(global_relative, 1e6)

    # Stable bilateral edge ranking.
    order = np.argsort(effective_d, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    positions = np.arange(n, dtype=float)[None, :]
    np.put_along_axis(
        rank,
        order,
        np.broadcast_to(positions, (n, n)),
        axis=1,
    )

    rank_denominator = np.maximum(degree - 1.0, 1.0)
    normalized_rank = rank / rank_denominator[:, None]
    normalized_rank = np.where(valid, np.minimum(normalized_rank, 1.0), 1.0)

    bilateral_rank = np.sqrt(
        np.maximum(1.0 - normalized_rank, 0.0)
        * np.maximum(1.0 - normalized_rank.T, 0.0)
    )
    bilateral_rank = np.maximum(bilateral_rank, eps)

    nearest = np.argmin(effective_d, axis=1)
    has_neighbor = degree > 0.0
    node_ids = np.arange(n)

    mutual_nearest = (
        has_neighbor[:, None]
        & has_neighbor[None, :]
        & (nearest[:, None] == node_ids[None, :])
        & (nearest[None, :] == node_ids[:, None])
        & valid
    ).astype(float)

    # Local density and sparse-node temperature.
    density_radius = 1.6 * local_scale
    local_density = np.mean(
        valid & (effective_d <= density_radius[:, None]),
        axis=1,
    )

    excess_density = np.maximum(local_density - 0.40, 0.0)
    density_penalty = (
        excess_density[:, None] + excess_density[None, :]
    ) * np.minimum(relative_length, 3.0)

    sparsity = 1.0 / np.sqrt(
        np.maximum(degree_fraction[:, None], eps)
        * np.maximum(degree_fraction[None, :], eps)
    )
    sparsity = np.minimum(sparsity, 8.0)

    temperature = 0.80 + 0.45 * np.minimum(sparsity - 1.0, 3.0)
    length_penalty = np.minimum(relative_length / temperature, 18.0)

    scale_mismatch = np.abs(
        np.log(
            np.maximum(local_scale[:, None], eps)
            / np.maximum(local_scale[None, :], eps)
        )
    )
    scale_penalty = np.minimum(scale_mismatch, 10.0)

    reciprocal_preference = np.log(
        np.maximum(global_scale / effective_d, eps)
    )
    global_penalty = np.minimum(global_relative, 24.0)

    log_score = (
        1.08 * reciprocal_preference
        + 1.05 * np.log(bilateral_rank)
        - 0.82 * length_penalty
        - 0.20 * global_penalty
        + 0.34 * mutual_nearest
        - 0.07 * density_penalty
        - 0.08 * scale_penalty
        + 0.10 * np.log(sparsity)
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.exp(np.clip(log_score - maximum, -745.0, 0.0))
    score = np.where(valid, score, 0.0)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
