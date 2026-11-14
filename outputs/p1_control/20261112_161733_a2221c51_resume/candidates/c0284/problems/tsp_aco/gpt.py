import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-priority matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite_positive = np.isfinite(d) & (d > 0.0)
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = np.zeros_like(d, dtype=float)
    symmetric_d[valid] = 0.5 * (d[valid] + d.T[valid])

    edge_values = symmetric_d[valid]
    global_scale = max(float(np.median(edge_values)), eps)
    maximum_distance = max(float(np.max(edge_values)), global_scale)
    fallback_distance = max(10.0 * maximum_distance, global_scale)

    work = np.where(valid, symmetric_d, fallback_distance)
    np.fill_diagonal(work, fallback_distance)

    # Primary local scale: the second available neighbor when possible.
    kth = min(1, n - 2)
    local_scale = np.partition(work, kth=kth, axis=1)[:, kth]

    valid_counts = valid.sum(axis=1)
    missing_scale = (valid_counts <= kth)
    if np.any(missing_scale):
        row_medians = np.full(n, global_scale, dtype=float)
        for i in np.flatnonzero(missing_scale):
            row_values = symmetric_d[i, valid[i]]
            if row_values.size:
                row_medians[i] = float(np.median(row_values))
        local_scale = np.where(missing_scale, row_medians, local_scale)

    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = work / endpoint_scale
    global_length = work / global_scale

    endpoint_potential = (
        endpoint_scale / np.maximum(work, eps)
    )
    endpoint_potential = np.maximum(endpoint_potential, eps)

    # Continuous reciprocal ranking from both endpoints.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    positions = np.broadcast_to(
        np.arange(n, dtype=float)[None, :],
        (n, n),
    )
    np.put_along_axis(ranks, order, positions, axis=1)

    ranks[~valid] = float(n)
    rank_denominator = max(n - 1, 1)
    mutual_rank = (
        ranks / rank_denominator
        + ranks.T / rank_denominator
    )
    rank_affinity = np.exp(-1.25 * mutual_rank)
    rank_affinity = np.maximum(rank_affinity, eps)

    scale_mismatch = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_compatibility = np.exp(
        -0.35 * np.minimum(scale_mismatch, 50.0)
    )
    scale_compatibility = np.maximum(scale_compatibility, eps)

    endpoint_longness = np.maximum(
        work / local_scale[:, None],
        work / local_scale[None, :],
    )
    excess_longness = np.maximum(endpoint_longness - 1.0, 0.0)

    global_preference = (
        0.65 + 0.35 * np.exp(-0.20 * global_length)
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_score = (
            0.95 * np.log(endpoint_potential)
            + 1.30 * np.log(rank_affinity)
            + 0.40 * np.log(scale_compatibility)
            + np.log(np.maximum(global_preference, eps))
            - 0.55 * relative_length
            - 0.25 * excess_longness
        )

    finite_scores = valid & np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)

    with np.errstate(over="ignore", invalid="ignore"):
        score[finite_scores] = np.exp(
            np.clip(
                log_score[finite_scores] - maximum,
                -745.0,
                0.0,
            )
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
