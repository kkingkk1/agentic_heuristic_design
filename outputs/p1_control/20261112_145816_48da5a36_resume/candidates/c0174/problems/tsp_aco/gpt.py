import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    directed_valid = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(directed_valid, False)

    if not np.any(directed_valid):
        return np.zeros((n, n), dtype=float)

    values = np.where(directed_valid, distances, 0.0)
    valid_count = directed_valid.astype(float) + directed_valid.T.astype(float)
    symmetric_sum = values + values.T

    distances = np.divide(
        symmetric_sum,
        valid_count,
        out=np.zeros_like(symmetric_sum),
        where=valid_count > 0.0,
    )

    valid = (valid_count > 0.0) & np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = distances[valid]
    median_distance = max(float(np.median(positive_values)), eps)
    maximum_distance = max(float(np.max(positive_values)), median_distance)
    replacement = max(10.0 * maximum_distance, median_distance)

    work = np.where(valid, distances, replacement)
    np.fill_diagonal(work, replacement)

    sorted_work = np.sort(work, axis=1)
    scale_rank = min(2, n - 2)
    local_scale = np.maximum(sorted_work[:, scale_rank], eps)

    normalized = work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized = np.maximum(normalized, 0.0)

    proximity = np.exp(-np.minimum(normalized, 50.0))

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)

    rank_temperature = 1.35
    rank_score = np.exp(-ranks / rank_temperature)
    rank_score /= np.maximum(
        np.max(rank_score, axis=1, keepdims=True),
        eps,
    )
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    log_scales = np.log(local_scale)
    scale_gap = np.abs(log_scales[:, None] - log_scales[None, :])
    scale_compatibility = np.exp(-0.5 * np.minimum(scale_gap, 50.0))

    neighborhood_kernel = np.exp(
        -np.minimum(work / local_scale[:, None], 50.0)
    ) * valid
    neighborhood_mass = np.sum(neighborhood_kernel, axis=1)

    exclusivity = 1.0 / np.sqrt(
        (1.0 + neighborhood_mass[:, None])
        * (1.0 + neighborhood_mass[None, :])
    )

    candidate_count = max(2, min(5, n - 1))
    candidate_threshold = np.take_along_axis(
        sorted_work,
        np.full((n, 1), candidate_count - 1, dtype=int),
        axis=1,
    )
    candidate_threshold = candidate_threshold[:, 0]

    relative_to_candidate = work / np.maximum(
        np.sqrt(candidate_threshold[:, None] * candidate_threshold[None, :]),
        eps,
    )
    soft_candidate_gate = 1.0 / (
        1.0 + np.exp(np.clip(3.0 * (relative_to_candidate - 1.0), -60.0, 60.0))
    )

    log_score = (
        1.05 * np.log(np.maximum(proximity, eps))
        + 1.10 * np.log(np.maximum(mutual_rank, eps))
        + 0.55 * np.log(np.maximum(scale_compatibility, eps))
        + 0.40 * np.log(np.maximum(exclusivity, eps))
        + 0.65 * np.log(np.maximum(soft_candidate_gate, eps))
    )

    valid_log_score = log_score[valid]
    if valid_log_score.size == 0:
        return np.zeros((n, n), dtype=float)

    log_score -= np.max(valid_log_score)
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
