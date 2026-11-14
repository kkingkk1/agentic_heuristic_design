import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d) & (d >= 0.0)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    eps = np.finfo(float).eps

    positive = d[valid & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    work = np.where(valid, np.maximum(d, 0.0), np.inf)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_work = np.take_along_axis(work, order, axis=1)

    row_index = np.arange(n)
    valid_count = np.sum(valid, axis=1)
    positive_count = np.sum(valid & (work > 0.0), axis=1)
    zero_count = valid_count - positive_count

    second_index = np.maximum(np.minimum(positive_count, 2) - 1, 0)
    scale_index = np.minimum(zero_count + second_index, n - 1)
    local_scale = sorted_work[row_index, scale_index]

    fallback_scale = np.min(work, axis=1)
    local_scale = np.where(
        positive_count > 0,
        local_scale,
        np.where(np.isfinite(fallback_scale), fallback_scale, global_scale),
    )
    local_scale = np.maximum(local_scale, eps)

    normalized_length = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = np.where(valid, normalized_length, np.inf)
    normalized_length = np.minimum(normalized_length, 1e6)

    rank_positions = np.empty((n, n), dtype=float)
    rank_positions[row_index[:, None], order] = np.arange(n, dtype=float)

    rank_denominator = np.maximum(valid_count - 1, 1)
    ranks = rank_positions / rank_denominator[:, None]
    ranks = np.where(valid, np.minimum(ranks, 1.0), 1.0)

    bilateral_rank_quality = np.sqrt(
        np.maximum(1.0 - ranks, 0.0)
        * np.maximum(1.0 - ranks.T, 0.0)
    )
    bilateral_rank_quality = np.maximum(bilateral_rank_quality, eps)

    reciprocal = global_scale / np.maximum(d, eps)
    reciprocal = np.where(valid, np.minimum(reciprocal, 1e12), 0.0)

    nearest = np.full(n, -1, dtype=int)
    has_neighbor = valid_count > 0
    nearest[has_neighbor] = order[has_neighbor, 0]

    mutual_nearest = (
        has_neighbor[:, None]
        & has_neighbor[None, :]
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    density_radius = 1.5 * local_scale
    density_counts = np.sum(
        sorted_work <= density_radius[:, None],
        axis=1,
        dtype=float,
    )
    local_density = density_counts / float(n)

    excess_density = np.maximum(local_density - 0.40, 0.0)
    density_penalty = (
        excess_density[:, None] + excess_density[None, :]
    ) * np.minimum(normalized_length, 3.0)

    scale_mismatch = np.abs(
        np.log(
            np.maximum(local_scale[:, None], eps)
            / np.maximum(local_scale[None, :], eps)
        )
    )
    scale_penalty = np.minimum(scale_mismatch, 12.0)

    log_score = (
        1.00 * np.log(np.maximum(reciprocal, eps))
        + 0.92 * np.log(bilateral_rank_quality)
        - 0.78 * normalized_length
        + 0.32 * mutual_nearest
        - 0.10 * scale_penalty
        - 0.06 * density_penalty
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.exp(np.clip(log_score - maximum, -745.0, 0.0))
    score = np.where(valid, score, 0.0)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
