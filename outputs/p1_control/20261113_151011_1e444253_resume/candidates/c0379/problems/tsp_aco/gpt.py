import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (raw + raw.T)
    eps = np.finfo(float).eps

    work = np.where(valid, d, np.inf)
    positive = work[np.isfinite(work) & (work > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    row_index = np.arange(n)
    valid_count = np.sum(valid, axis=1)

    # Reuse stable distance rankings for all rank-based signals.
    order = np.argsort(work, axis=1, kind="stable")

    rank_positions = np.empty((n, n), dtype=float)
    rank_positions[row_index[:, None], order] = np.arange(n, dtype=float)

    rank_denominator = np.maximum(valid_count - 1, 1)
    ranks = rank_positions / rank_denominator[:, None]
    ranks = np.where(valid, np.clip(ranks, 0.0, 1.0), 1.0)

    bilateral_rank_quality = np.sqrt(
        np.maximum(1.0 - ranks, 0.0)
        * np.maximum(1.0 - ranks.T, 0.0)
    )
    bilateral_rank_quality = np.maximum(bilateral_rank_quality, eps)

    # Use the second positive neighbor as a robust local scale.
    positive_work = np.where(valid & (work > 0.0), work, np.inf)
    positive_sorted = np.sort(positive_work, axis=1)
    positive_count = np.sum(np.isfinite(positive_work), axis=1)

    second_index = np.maximum(np.minimum(positive_count, 2) - 1, 0)
    local_scale = positive_sorted[row_index, second_index]

    nearest_valid = np.min(work, axis=1)
    local_scale = np.where(
        positive_count > 0,
        local_scale,
        np.where(np.isfinite(nearest_valid), nearest_valid, global_scale),
    )
    local_scale = np.maximum(local_scale, global_scale * 1e-12)

    relative_length = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = np.where(valid, relative_length, np.inf)
    relative_length = np.clip(relative_length, 0.0, 1e6)

    # Bounded reciprocal distance avoids numerical domination by zero edges.
    scaled_distance = np.maximum(d / global_scale, 0.0)
    reciprocal_quality = 1.0 / (scaled_distance + 0.025)
    reciprocal_quality = np.where(valid, reciprocal_quality, 0.0)

    nearest = np.full(n, -1, dtype=int)
    has_neighbor = valid_count > 0
    nearest[has_neighbor] = order[has_neighbor, 0]

    mutual_nearest = (
        has_neighbor[:, None]
        & has_neighbor[None, :]
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    # Continuous retention of promising neighborhoods; no hard edge cutoff.
    mean_rank = 0.5 * (ranks + ranks.T)
    neighborhood_retention = np.exp(-np.square(mean_rank / 0.38))

    density_radius = 1.5 * local_scale
    local_density = np.mean(
        valid & (work <= density_radius[:, None]),
        axis=1,
    )
    excess_density = np.maximum(local_density - 0.45, 0.0)
    density_penalty = (
        excess_density[:, None] + excess_density[None, :]
    ) * np.minimum(relative_length, 3.0)

    log_score = (
        1.05 * np.log(np.maximum(reciprocal_quality, eps))
        + 1.00 * np.log(bilateral_rank_quality)
        - 0.55 * relative_length
        + 0.24 * mutual_nearest
        + 0.20 * neighborhood_retention
        - 0.035 * density_penalty
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
