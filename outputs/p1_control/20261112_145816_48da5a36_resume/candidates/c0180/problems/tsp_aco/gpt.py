import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    distances = 0.5 * (distances + distances.T)
    valid = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[valid]
    median_distance = max(float(np.median(positive_values)), eps)
    maximum_distance = max(float(np.max(positive_values)), median_distance)
    replacement = max(8.0 * maximum_distance, 2.0 * median_distance)

    work = np.where(valid, distances, replacement)
    np.fill_diagonal(work, replacement)

    neighbor_rank = min(1, n - 2)
    sorted_work = np.sort(work, axis=1)
    local_scale = np.maximum(sorted_work[:, neighbor_rank], eps)

    relative_length = work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = np.maximum(relative_length, 0.0)

    proximity = np.exp(-np.minimum(1.15 * relative_length, 50.0))

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)

    rank_score = 1.0 / np.power(1.0 + ranks, 0.9)
    rank_score /= np.maximum(
        np.max(rank_score, axis=1, keepdims=True),
        eps,
    )
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    log_scale = np.log(np.maximum(local_scale, eps))
    scale_gap = np.abs(log_scale[:, None] - log_scale[None, :])
    scale_compatibility = np.exp(
        -0.28 * np.minimum(scale_gap, 50.0)
    )

    neighborhood_mass = np.sum(
        np.exp(-np.minimum(work / local_scale[:, None], 50.0)) * valid,
        axis=1,
    )
    crowding_penalty = 1.0 / np.sqrt(
        (1.0 + 0.75 * neighborhood_mass[:, None])
        * (1.0 + 0.75 * neighborhood_mass[None, :])
    )

    log_score = (
        0.95 * np.log(np.maximum(proximity, eps))
        + 1.05 * np.log(np.maximum(mutual_rank, eps))
        + 0.40 * np.log(np.maximum(scale_compatibility, eps))
        + 0.22 * np.log(np.maximum(crowding_penalty, eps))
    )

    valid_scores = log_score[valid]
    if valid_scores.size == 0:
        return np.zeros((n, n), dtype=float)

    log_score -= np.max(valid_scores)
    score = np.exp(np.clip(log_score, -745.0, 0.0))

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
