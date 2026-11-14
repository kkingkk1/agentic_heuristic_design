import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d > 0.0)
    valid &= valid.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = np.zeros((n, n), dtype=float)
    symmetric_d[valid] = 0.5 * (d[valid] + d.T[valid])

    positive_values = symmetric_d[valid]
    eps = np.finfo(float).eps
    global_scale = max(float(np.median(positive_values)), eps)
    maximum_distance = max(float(np.max(positive_values)), global_scale)
    padding = max(10.0 * maximum_distance, global_scale)

    work = np.where(valid, symmetric_d, padding)
    np.fill_diagonal(work, padding)

    degree = valid.sum(axis=1)
    local_scale = np.full(n, global_scale, dtype=float)
    nearest_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row = symmetric_d[i, valid[i]]
        if row.size == 0:
            continue

        row = np.sort(row)
        nearest_scale[i] = max(float(row[0]), eps)

        target_rank = min(2, row.size - 1)
        local_scale[i] = max(float(row[target_rank]), eps)

    density_scale = np.sqrt(
        np.maximum(local_scale, eps) * np.maximum(nearest_scale, eps)
    )

    edge_scale = np.sqrt(
        np.maximum(local_scale[:, None], eps)
        * np.maximum(local_scale[None, :], eps)
    )

    relative_length = symmetric_d / np.maximum(edge_scale, eps)
    global_length = symmetric_d / global_scale

    endpoint_potential = np.sqrt(
        np.maximum(local_scale[:, None], eps)
        / np.maximum(symmetric_d, eps)
    ) * np.sqrt(
        np.maximum(local_scale[None, :], eps)
        / np.maximum(symmetric_d, eps)
    )

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    row_index = np.arange(n)[:, None]
    ranks[row_index, order] = np.arange(n, dtype=float)[None, :]
    ranks[~valid] = float(n + 1)

    rank_denominator = float(max(n - 1, 1))
    rank_mean = 0.5 * (ranks + ranks.T) / rank_denominator
    rank_affinity = np.exp(-1.05 * np.minimum(rank_mean, 4.0))

    candidate_width = min(max(4, n // 5), max(n - 1, 1))
    left_candidate = ranks < candidate_width
    right_candidate = left_candidate.T

    mutual_support = (left_candidate & right_candidate).astype(float)
    one_sided_support = (left_candidate | right_candidate).astype(float)

    neighborhood_support = (
        1.0
        + 0.70 * mutual_support
        + 0.10 * one_sided_support
    )

    log_density_ratio = np.abs(
        np.log(np.maximum(density_scale[:, None], eps))
        - np.log(np.maximum(density_scale[None, :], eps))
    )
    density_compatibility = np.exp(
        -0.22 * np.minimum(log_density_ratio, 50.0)
    )

    endpoint_longness = np.maximum(
        symmetric_d / np.maximum(local_scale[:, None], eps),
        symmetric_d / np.maximum(local_scale[None, :], eps),
    )
    excess_longness = np.maximum(endpoint_longness - 1.10, 0.0)

    node_competition = np.sqrt(
        np.maximum(local_scale[:, None], eps)
        / np.maximum(symmetric_d, eps)
    ) * np.sqrt(
        np.maximum(local_scale[None, :], eps)
        / np.maximum(symmetric_d, eps)
    )
    node_competition = np.minimum(node_competition, 3.0)

    bridge_support = np.exp(
        -0.10 * np.maximum(relative_length - 1.0, 0.0)
    )

    degree_balance = np.sqrt(
        np.minimum(degree[:, None], degree[None, :]) + 1.0
    ) / np.sqrt(
        np.maximum(np.maximum(degree[:, None], degree[None, :]), 1.0)
    )

    log_score = (
        1.35 * np.log(np.maximum(endpoint_potential, eps))
        + 1.20 * np.log(np.maximum(rank_affinity, eps))
        + 0.62 * np.log(np.maximum(neighborhood_support, eps))
        + 0.32 * np.log(np.maximum(density_compatibility, eps))
        + 0.16 * np.log(np.maximum(node_competition, eps))
        + 0.10 * np.log(np.maximum(bridge_support, eps))
        + 0.08 * np.log(np.maximum(degree_balance, eps))
        - 0.34 * relative_length
        - 0.055 * global_length
        - 0.24 * excess_longness
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
