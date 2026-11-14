import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-priority matrix for stochastic TSP sampling."""
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

    symmetric_distance = 0.5 * (d + d.T)
    positive = valid & (symmetric_distance > 0.0)

    positive_values = symmetric_distance[positive]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    work = np.where(valid, np.maximum(symmetric_distance, 0.0), np.inf)
    positive_work = np.where(positive, symmetric_distance, np.inf)

    positive_sorted = np.sort(positive_work, axis=1)
    positive_count = positive.sum(axis=1)

    second_index = np.minimum(positive_count, 2) - 1
    second_index = np.maximum(second_index, 0)
    rows = np.arange(n)

    local_scale = positive_sorted[rows, second_index]
    nearest_scale = np.min(work, axis=1)

    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        np.where(
            np.isfinite(nearest_scale),
            nearest_scale,
            global_scale,
        ),
    )
    local_scale = np.maximum(local_scale, eps)

    maximum_distance = (
        float(np.max(positive_values))
        if positive_values.size
        else global_scale
    )
    fallback_distance = max(maximum_distance, global_scale)
    safe_distance = np.where(
        np.isfinite(work),
        work,
        10.0 * fallback_distance,
    )
    safe_distance = np.maximum(safe_distance, eps)

    edge_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = safe_distance / edge_scale
    global_length = safe_distance / global_scale

    order = np.argsort(safe_distance, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[rows[:, None], order] = np.arange(n, dtype=float)[None, :]
    rank /= float(max(n - 1, 1))

    bilateral_rank = np.exp(-3.0 * (rank + rank.T))

    candidate_width = min(max(3, n // 5), max(n - 1, 1))
    mutual_candidate = (
        (rank < candidate_width) & (rank.T < candidate_width)
    ).astype(float)
    one_sided_candidate = (
        (rank < candidate_width) | (rank.T < candidate_width)
    ).astype(float)
    neighborhood_support = (
        1.0
        + 0.45 * mutual_candidate
        + 0.10 * one_sided_candidate
    )

    endpoint_potential = np.sqrt(
        np.maximum(local_scale[:, None], eps)
        / safe_distance
        * np.maximum(local_scale[None, :], eps)
        / safe_distance
    )

    scale_difference = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_balance = np.exp(-0.25 * np.minimum(scale_difference, 50.0))

    endpoint_longness = np.maximum(
        safe_distance / np.maximum(local_scale[:, None], eps),
        safe_distance / np.maximum(local_scale[None, :], eps),
    )
    excess_longness = np.maximum(endpoint_longness - 1.15, 0.0)

    global_preference = 0.62 + 0.38 * np.exp(
        -0.50 * global_length
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_score = (
            1.10 * np.log(np.maximum(endpoint_potential, eps))
            + 1.35 * np.log(np.maximum(bilateral_rank, eps))
            + 0.50 * np.log(np.maximum(neighborhood_support, eps))
            + 0.35 * np.log(np.maximum(scale_balance, eps))
            + 1.00 * np.log(np.maximum(global_preference, eps))
            - 0.42 * relative_length
            - 0.18 * excess_longness
        )

    finite_scores = valid & np.isfinite(log_score)
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
