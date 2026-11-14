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

    symmetric_d = 0.5 * (d + d.T)
    symmetric_d = np.where(valid, np.maximum(symmetric_d, 0.0), np.inf)

    eps = np.finfo(float).eps

    positive_mask = valid & (symmetric_d > 0.0)
    positive_values = symmetric_d[positive_mask]

    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    positive_work = np.where(positive_mask, symmetric_d, np.inf)
    first_scale = np.min(positive_work, axis=1)

    if n > 1:
        second_index = min(1, n - 1)
        second_scale = np.partition(
            positive_work,
            second_index,
            axis=1,
        )[:, second_index]
    else:
        second_scale = first_scale.copy()

    local_scale = np.where(
        np.isfinite(second_scale),
        second_scale,
        np.where(
            np.isfinite(first_scale),
            first_scale,
            global_scale,
        ),
    )
    local_scale = np.maximum(local_scale, eps)

    work = np.where(valid, symmetric_d, np.inf)
    normalized_length = work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = np.where(valid, normalized_length, np.inf)
    normalized_length = np.minimum(normalized_length, 1e6)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    valid_count = valid.sum(axis=1).astype(float)
    denominator = np.maximum(valid_count - 1.0, 1.0)
    ranks = ranks / denominator[:, None]
    ranks = np.where(valid, np.minimum(ranks, 1.0), 1.0)

    bilateral_rank = np.sqrt(
        np.maximum(1.0 - ranks, 0.0)
        * np.maximum(1.0 - ranks.T, 0.0)
    )
    bilateral_rank = np.maximum(bilateral_rank, eps)

    candidate_width = np.minimum(
        np.maximum(3, n // 5),
        np.maximum(valid_count[:, None], 1.0),
    )
    mutual_candidate = (
        (ranks < candidate_width)
        & (ranks.T < candidate_width.T)
        & valid
    ).astype(float)

    one_sided_candidate = (
        ((ranks < candidate_width) | (ranks.T < candidate_width.T))
        & valid
    ).astype(float)

    neighborhood_support = (
        1.0
        + 0.42 * mutual_candidate
        + 0.08 * one_sided_candidate
    )

    reciprocal = global_scale / np.maximum(symmetric_d, eps)
    reciprocal = np.where(valid, np.minimum(reciprocal, 1e12), 0.0)

    density_radius = 1.55 * local_scale
    local_density = np.mean(
        valid & (work <= density_radius[:, None]),
        axis=1,
    )

    excess_density = np.maximum(local_density - 0.45, 0.0)
    density_penalty = (
        excess_density[:, None]
        + excess_density[None, :]
    ) * np.minimum(normalized_length, 3.0)

    scale_mismatch = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_penalty = np.minimum(scale_mismatch, 12.0)

    endpoint_longness = np.maximum(
        normalized_length,
        normalized_length.T,
    )
    excess_longness = np.maximum(endpoint_longness - 1.20, 0.0)

    log_score = (
        1.05 * np.log(np.maximum(reciprocal, eps))
        + 1.02 * np.log(bilateral_rank)
        + 0.34 * np.log(np.maximum(neighborhood_support, eps))
        - 0.62 * normalized_length
        - 0.10 * scale_penalty
        - 0.055 * density_penalty
        - 0.12 * excess_longness
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
    score = np.maximum(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
