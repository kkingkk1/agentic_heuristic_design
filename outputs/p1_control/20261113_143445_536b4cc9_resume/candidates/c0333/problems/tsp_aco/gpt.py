import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
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

    symmetric_d = 0.5 * (d + d.T)
    symmetric_d = np.maximum(symmetric_d, 0.0)

    work = np.where(valid, symmetric_d, np.inf)

    positive_valid = valid & (symmetric_d > 0.0)
    positive_work = np.where(positive_valid, symmetric_d, np.inf)
    positive_count = np.sum(positive_valid, axis=1)

    positive_values = symmetric_d[positive_valid]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    sorted_positive = np.sort(positive_work, axis=1)
    second_index = np.maximum(np.minimum(positive_count, 2) - 1, 0)
    rows = np.arange(n)

    local_scale = sorted_positive[rows, second_index]
    nearest_scale = np.min(work, axis=1)

    local_scale = np.where(
        positive_count > 0,
        local_scale,
        np.where(np.isfinite(nearest_scale), nearest_scale, global_scale),
    )
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = np.divide(
        work,
        pair_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(work),
    )
    relative_length = np.minimum(relative_length, 1e6)

    global_length = np.divide(
        work,
        global_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(work),
    )
    global_length = np.minimum(global_length, 1e6)

    order = np.argsort(work, axis=1, kind="stable")
    rank_positions = np.empty((n, n), dtype=float)
    rank_positions[rows[:, None], order] = np.arange(n, dtype=float)[None, :]

    valid_count = np.sum(valid, axis=1)
    rank_denominator = np.maximum(valid_count - 1, 1)

    normalized_rank = rank_positions / rank_denominator[:, None]
    normalized_rank = np.where(
        valid,
        np.minimum(normalized_rank, 1.0),
        1.0,
    )

    bilateral_rank = np.sqrt(
        np.maximum(1.0 - normalized_rank, 0.0)
        * np.maximum(1.0 - normalized_rank.T, 0.0)
    )
    bilateral_rank = np.maximum(bilateral_rank, eps)

    reciprocal = np.divide(
        global_scale,
        np.maximum(work, eps),
        out=np.zeros((n, n), dtype=float),
        where=np.isfinite(work),
    )
    reciprocal = np.clip(reciprocal, eps, 1e12)

    endpoint_ratio = np.maximum(
        np.divide(
            work,
            local_scale[:, None],
            out=np.full((n, n), np.inf, dtype=float),
            where=np.isfinite(work),
        ),
        np.divide(
            work,
            local_scale[None, :],
            out=np.full((n, n), np.inf, dtype=float),
            where=np.isfinite(work),
        ),
    )

    compatibility = np.exp(-np.minimum(endpoint_ratio, 30.0))
    compatibility = np.maximum(compatibility, eps)

    candidate_limit = 1.75 * np.maximum(
        local_scale[:, None],
        local_scale[None, :],
    )
    mutual_candidate = (
        valid
        & (work <= candidate_limit)
        & (work.T <= candidate_limit)
    ).astype(float)

    density_radius = 1.5 * local_scale[:, None]
    local_density = np.mean(
        valid & (work <= density_radius),
        axis=1,
    )
    excess_density = np.maximum(local_density - 0.35, 0.0)
    density_penalty = (
        excess_density[:, None] + excess_density[None, :]
    ) * np.minimum(relative_length, 3.0)

    scale_mismatch = np.abs(
        np.log(
            np.maximum(local_scale[:, None], eps)
            / np.maximum(local_scale[None, :], eps)
        )
    )
    scale_penalty = np.minimum(scale_mismatch, 12.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_score = (
            1.00 * np.log(np.maximum(reciprocal, eps))
            + 1.05 * np.log(bilateral_rank)
            - 0.78 * np.minimum(relative_length, 12.0)
            - 0.12 * np.minimum(global_length, 20.0)
            + 0.30 * mutual_candidate
            + 0.38 * np.log(compatibility)
            - 0.08 * scale_penalty
            - 0.07 * density_penalty
        )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    shift = float(np.max(log_score[finite_scores]))

    with np.errstate(over="ignore", invalid="ignore"):
        score = np.exp(
            np.clip(log_score - shift, -745.0, 0.0)
        )

    score = np.where(valid, score, 0.0)
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
