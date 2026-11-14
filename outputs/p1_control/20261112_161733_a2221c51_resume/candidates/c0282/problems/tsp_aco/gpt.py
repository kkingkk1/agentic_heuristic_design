import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
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

    symmetric_d = 0.5 * (d + d.T)
    symmetric_d = np.where(valid, symmetric_d, np.inf)

    edge_values = symmetric_d[valid]
    global_scale = max(float(np.median(edge_values)), eps)
    fallback_distance = max(float(np.max(edge_values)), global_scale)

    work = np.where(valid, symmetric_d, fallback_distance * 10.0)
    np.fill_diagonal(work, fallback_distance * 10.0)

    neighborhood_rank = min(1, n - 2)
    local_scale = np.partition(
        work,
        kth=neighborhood_rank,
        axis=1,
    )[:, neighborhood_rank]

    finite_local = np.isfinite(local_scale)
    local_scale = np.where(
        finite_local,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = work / endpoint_scale
    global_length = work / global_scale

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    valid_count = valid.sum(axis=1).astype(float)
    rank_denominator = np.maximum(valid_count - 1.0, 1.0)
    normalized_ranks = ranks / rank_denominator[:, None]
    normalized_ranks = np.minimum(normalized_ranks, 1.0)

    mutual_rank_quality = np.sqrt(
        np.maximum(1.0 - normalized_ranks, 0.0)
        * np.maximum(1.0 - normalized_ranks.T, 0.0)
    )
    mutual_rank_quality = np.maximum(mutual_rank_quality, eps)

    scale_difference = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_compatibility = np.exp(
        -0.35 * np.minimum(scale_difference, 50.0)
    )

    raw_global_preference = np.exp(
        -0.26 * global_length
    )
    global_preference = 0.68 + 0.32 * raw_global_preference

    endpoint_longness = np.maximum(
        work / local_scale[:, None],
        work / local_scale[None, :],
    )
    excess_longness = np.maximum(endpoint_longness - 1.0, 0.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_score = (
            0.95 * np.log(
                np.maximum(local_scale[:, None], eps)
                / np.maximum(work, eps)
            )
            + 1.20 * np.log(mutual_rank_quality)
            + 0.45 * np.log(np.maximum(scale_compatibility, eps))
            + np.log(np.maximum(global_preference, eps))
            - 0.55 * relative_length
            - 0.18 * excess_longness
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
