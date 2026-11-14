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

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    d = np.maximum(d, 0.0)

    positive = valid & (d > 0.0)
    positive_values = d[positive]

    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    work = np.where(valid, d, np.inf)
    positive_work = np.where(positive, d, np.inf)

    positive_count = np.sum(np.isfinite(positive_work), axis=1)
    local_index = np.maximum(np.minimum(positive_count, 2) - 1, 0)

    local_candidates = np.partition(
        positive_work,
        kth=min(1, n - 1),
        axis=1,
    )
    local_scale = local_candidates[
        np.arange(n),
        local_index,
    ]

    nearest_valid = np.min(work, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        np.where(np.isfinite(nearest_valid), nearest_valid, global_scale),
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = work / np.maximum(pair_scale, eps)
    relative_length = np.minimum(relative_length, 1e6)

    global_length = work / global_scale
    global_length = np.minimum(global_length, 1e6)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    valid_counts = np.sum(valid, axis=1)
    rank_denominator = np.maximum(valid_counts - 1, 1)
    ranks = ranks / rank_denominator[:, None]
    ranks = np.where(valid, np.minimum(ranks, 1.0), 1.0)

    bilateral_rank = np.sqrt(
        np.maximum(1.0 - ranks, 0.0)
        * np.maximum(1.0 - ranks.T, 0.0)
    )
    bilateral_rank = np.maximum(bilateral_rank, eps)

    reciprocal = global_scale / np.maximum(work, eps)
    reciprocal = np.minimum(reciprocal, 1e12)
    reciprocal = np.where(valid, reciprocal, eps)

    nearest = np.full(n, -1, dtype=int)
    has_neighbor = valid_counts > 0
    nearest[has_neighbor] = order[has_neighbor, 0]

    node_ids = np.arange(n)
    mutual_nearest = (
        has_neighbor[:, None]
        & has_neighbor[None, :]
        & (nearest[:, None] == node_ids[None, :])
        & (nearest[None, :] == node_ids[:, None])
    ).astype(float)

    endpoint_ratio = np.maximum(
        work / local_scale[:, None],
        work / local_scale[None, :],
    )
    endpoint_ratio = np.minimum(endpoint_ratio, 50.0)

    local_compatibility = np.exp(-0.55 * endpoint_ratio)

    scale_mismatch = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_compatibility = np.exp(
        -0.18 * np.minimum(scale_mismatch, 50.0)
    )

    degree_fraction = valid_counts.astype(float) / max(n - 1, 1)
    degree_fraction = np.maximum(degree_fraction, 1.0 / max(n - 1, 1))
    sparsity_bonus = 1.0 / np.sqrt(
        degree_fraction[:, None] * degree_fraction[None, :]
    )
    sparsity_bonus = np.minimum(sparsity_bonus, 6.0)

    log_score = (
        0.85 * np.log(np.maximum(reciprocal, eps))
        + 1.10 * np.log(bilateral_rank)
        + 0.75 * np.log(np.maximum(local_compatibility, eps))
        + 0.30 * mutual_nearest
        + 0.35 * np.log(np.maximum(scale_compatibility, eps))
        + 0.08 * np.log(np.maximum(sparsity_bonus, eps))
        - 0.58 * relative_length
        - 0.18 * global_length
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - maximum, -745.0, 0.0)
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
