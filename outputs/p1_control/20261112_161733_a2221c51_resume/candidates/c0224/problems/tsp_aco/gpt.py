import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite = np.isfinite(d) & (d >= 0.0)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = 0.5 * (d + d.T)
    positive = valid & (symmetric_d > 0.0)

    positive_values = symmetric_d[positive]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    # Zero-length valid edges remain usable but are protected numerically.
    edge_distance = np.where(valid, np.maximum(symmetric_d, 0.0), np.inf)
    positive_distance = np.where(positive, symmetric_d, np.inf)

    local_scale = np.full(n, global_scale, dtype=float)
    for i in range(n):
        candidates = positive_distance[i]
        candidates = candidates[np.isfinite(candidates)]

        if candidates.size:
            local_scale[i] = candidates[min(1, candidates.size - 1)]
        else:
            finite_candidates = edge_distance[i][np.isfinite(edge_distance[i])]
            if finite_candidates.size:
                local_scale[i] = max(
                    float(np.min(finite_candidates)),
                    eps,
                )

    local_scale = np.maximum(local_scale, eps)

    safe_distance = np.maximum(edge_distance, eps)
    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = safe_distance / endpoint_scale
    relative_length = np.minimum(relative_length, 1e6)

    global_length = safe_distance / global_scale
    global_length = np.minimum(global_length, 1e6)

    # Continuous bilateral rank quality.
    ranks = np.ones((n, n), dtype=float)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[
                np.argsort(safe_distance[i, indices], kind="stable")
            ]
            denominator = max(order.size - 1, 1)
            ranks[i, order] = (
                np.arange(order.size, dtype=float) / denominator
            )

    bilateral_rank = np.sqrt(
        np.maximum(1.0 - ranks, eps)
        * np.maximum(1.0 - ranks.T, eps)
    )

    rank_agreement = np.exp(
        -1.5 * np.abs(ranks - ranks.T)
    )

    reciprocal = global_scale / safe_distance
    reciprocal = np.minimum(reciprocal, 1e12)

    # Soft mutual-nearest-neighbor reward.
    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            nearest[i] = indices[
                np.argmin(safe_distance[i, indices])
            ]

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    # Mildly penalize edges that are long for both endpoints.
    endpoint_longness = np.maximum(
        safe_distance / local_scale[:, None],
        safe_distance / local_scale[None, :],
    )
    excess_longness = np.maximum(endpoint_longness - 1.0, 0.0)
    excess_longness = np.minimum(excess_longness, 8.0)

    scale_mismatch = np.abs(
        np.log(local_scale[:, None])
        - np.log(local_scale[None, :])
    )
    scale_compatibility = np.exp(
        -0.35 * np.minimum(scale_mismatch, 20.0)
    )

    log_score = (
        0.82 * np.log(np.maximum(reciprocal, eps))
        + 0.88 * np.log(np.maximum(bilateral_rank, eps))
        + 0.42 * np.log(np.maximum(rank_agreement, eps))
        + 0.30 * np.log(np.maximum(scale_compatibility, eps))
        - 0.72 * relative_length
        - 0.14 * global_length
        - 0.24 * excess_longness
        + 0.22 * mutual_nearest
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
