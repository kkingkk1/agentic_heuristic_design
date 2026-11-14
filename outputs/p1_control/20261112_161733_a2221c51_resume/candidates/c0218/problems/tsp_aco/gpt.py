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

    d = 0.5 * (d + d.T)
    eps = np.finfo(float).eps

    positive_values = d[valid & (d > 0.0)]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    rank_work = np.where(valid, np.maximum(d, 0.0), np.inf)
    positive_work = np.where(valid & (d > 0.0), d, np.inf)

    positive_count = np.sum(np.isfinite(positive_work), axis=1)
    local_scale = np.full(n, global_scale, dtype=float)

    if n > 1:
        second_index = min(1, n - 2)
        nearest_two = np.partition(
            positive_work,
            kth=second_index,
            axis=1,
        )[:, : second_index + 1]

        for i in range(n):
            row = nearest_two[i]
            finite_row = row[np.isfinite(row)]
            if finite_row.size:
                local_scale[i] = finite_row[
                    min(1, finite_row.size - 1)
                ]
            elif positive_count[i] == 0:
                valid_row = d[i, valid[i]]
                positive_row = valid_row[valid_row > 0.0]
                if positive_row.size:
                    local_scale[i] = float(np.median(positive_row))

    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = rank_work / geometric_scale
    normalized_length = np.minimum(normalized_length, 1e6)

    absolute_length = rank_work / global_scale
    absolute_length = np.minimum(absolute_length, 1e6)

    order = np.argsort(rank_work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)[None, :]
    ranks /= max(n - 1, 1)
    ranks = np.where(valid, ranks, 1.0)

    bilateral_rank_quality = np.sqrt(
        np.maximum(1.0 - ranks, eps)
        * np.maximum(1.0 - ranks.T, eps)
    )

    reciprocal_distance = global_scale / np.maximum(rank_work, eps)
    reciprocal_distance = np.minimum(reciprocal_distance, 1e12)

    nearest = np.argmin(rank_work, axis=1)
    has_nearest = np.any(valid, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    for i in np.flatnonzero(has_nearest):
        j = int(nearest[i])
        if has_nearest[j] and nearest[j] == i:
            mutual_nearest[i, j] = 1.0

    neighborhood_radius = 1.5 * local_scale
    local_density = np.mean(
        valid & (rank_work <= neighborhood_radius[:, None]),
        axis=1,
    )
    excess_density = np.maximum(local_density - 0.35, 0.0)
    density_penalty = (
        excess_density[:, None] + excess_density[None, :]
    ) * np.minimum(normalized_length, 3.0)

    scale_gap = np.abs(
        np.log(
            np.maximum(local_scale[:, None], eps)
            / np.maximum(local_scale[None, :], eps)
        )
    )
    scale_gap = np.minimum(scale_gap, 12.0)

    log_score = (
        0.92 * np.log(np.maximum(reciprocal_distance, eps))
        + 0.86 * np.log(
            np.maximum(bilateral_rank_quality, eps)
        )
        - 0.82 * normalized_length
        - 0.18 * absolute_length
        + 0.24 * mutual_nearest
        - 0.10 * scale_gap
        - 0.08 * density_penalty
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.exp(np.clip(log_score - maximum, -745.0, 0.0))
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
